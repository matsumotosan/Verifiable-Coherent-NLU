import os
import pickle
import torch
import transformers

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

from www.model.transformers_ext import TieredModelPipeline
from www.model.eval import evaluate_tiered, save_results, save_preds, add_entity_attribute_labels
from www.model.train import train_epoch_tiered
from www.utils import print_dict, get_model_dir
from www.dataset.ann import att_to_idx, att_to_num_classes, att_types

from src.dataloading import get_dataloaders
from src.preprocessing import data_setup, get_baseline, get_tensor_dataset
from src.utils import get_components


def main(args):
    # Print run configuration
    print("====================")
    print("JOB CONFIGURATION")
    print(f"Model: {args.model}")
    print(f"Objective: {args.objective}")
    print(f"Loss weights: {args.loss_weights}")
    print(f"gamma: {args.gamma}")
    print(f"alpha: {args.alpha}")
    print(f"lambda_const: {args.lambda_const}")
    print(f"p_th: {args.p_th}")
    print(f"Batch size: {args.batch_size}")
    print(f"Eval batch size: {args.eval_batch_size}")
    print(f"Test batch size: {args.test_batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("====================")
    
    # Get model-related components (LM and tokenizer)
    model_name, model_class, config_class, emb_class, tokenizer, lm_class = get_components(args.model, args.cache_dir)

    # Preprocess data
    print('Preprocessing data.')
    cloze_dataset_2s, order_dataset_2s = data_setup()
    # print('here')
    tiered_dataset = get_baseline(cloze_dataset_2s, tokenizer)
    # print('here now')
    tiered_tensor_dataset = get_tensor_dataset(tiered_dataset)

    # Create dataloaders for train, val, and test datasets
    print('Getting dataloaders.')
    train_dataloader, dev_dataloader, test_dataloader = get_dataloaders(args, tiered_tensor_dataset)
    dev_dataset_name = args.subtask + '_%s_dev'
    dev_ids = [ex['example_id'] for ex in tiered_dataset['dev']]
    
    # Set number of state variables
    num_state_labels = {}
    for att in att_to_idx:
        if att_types[att] == 'default':
            num_state_labels[att_to_idx[att]] = 3
        else:
            num_state_labels[att_to_idx[att]] = att_to_num_classes[att]

    # Set up model
    config = config_class.from_pretrained(
        model_name,
        cache_dir=args.cache_dir
    )
    
    # Set up embedding
    emb = emb_class.from_pretrained(
        model_name,
        config=config,
        cache_dir=args.cache_dir
    )
    
    if torch.cuda.is_available():
        emb.cuda()

    device = emb.device
    
    max_story_length = max([len(ex['stories'][0]['sentences']) for p in tiered_dataset for ex in tiered_dataset[p]])
    
    # Initialize model
    print('Initializing model.')
    model = TieredModelPipeline(
        emb,
        max_story_length,
        len(att_to_num_classes),
        num_state_labels,
        config_class,
        model_name,
        device,
        ablation=args.ablation,
        objective=args.objective,
        loss_weights=args.loss_weights,
        gamma=args.gamma,
        alpha=args.alpha,
        lambda_const=args.lambda_const,
        p_th=args.p_th
    ).to(device)

    # Initialize optimizer and scheduler
    print('Initializing optimizer and scheduler.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs
    )

    # Initialize variables to track
    train_lc_data = []
    val_lc_data = []
    loss_values = []
    obj_values = []

    # Train model
    print('Training model')
    for epoch in range(args.num_epochs):
        train_loss, _ = train_epoch_tiered(
            model,
            optimizer,
            train_dataloader,
            device,
            epoch=epoch,
            seg_mode=False,
            build_learning_curves=args.generate_learning_curve,
            val_dataloader=dev_dataloader,
            train_lc_data=train_lc_data,
            val_lc_data=val_lc_data,
            grad_surgery=args.grad_surgery
        )
        
        loss_values.append(train_loss)

        # Validate on dev set
        validation_results = evaluate_tiered(
            model,
            dev_dataloader,
            device,
            [(accuracy_score, 'accuracy'), (f1_score, 'f1')],
            epoch,
            seg_mode=False,
            return_explanations=True
        )
        
        metr_attr, all_pred_atts, all_atts, \
        metr_prec, all_pred_prec, all_prec, \
        metr_eff, all_pred_eff, all_eff, \
        metr_conflicts, all_pred_conflicts, all_conflicts, \
        metr_stories, all_pred_stories, all_stories, explanations = validation_results[:16]
        explanations = add_entity_attribute_labels(explanations, tiered_dataset['dev'], list(att_to_num_classes.keys()))

        # Print results
        print(f"Epoch ({epoch + 1}/{args.num_epochs} -- train_loss: {train_loss}, val_loss: {0}")
        print('[%s] Validation results:' % str(epoch))
        print('[%s] Preconditions:' % str(epoch))
        print_dict(metr_prec)
        print('[%s] Effects:' % str(epoch))
        print_dict(metr_eff)
        print('[%s] Conflicts:' % str(epoch))
        print_dict(metr_conflicts)
        print('[%s] Stories:' % str(epoch))
        print_dict(metr_stories)

        # Save accuracy - want to maximize verifiability of tiered predictions
        ver = metr_stories['verifiability']
        # acc = metr_stories['accuracy']
        obj_values.append(ver)
        
        # Save model checkpoint
        print('[%s] Saving model checkpoint...' % str(epoch))
        model_dir = get_model_dir(
            model_name.replace('/', '-'),
            args.subtask,
            args.batch_size,
            args.learning_rate,
            epoch
        )
        model_param_str = model_dir + '_' +  '-'.join([str(lw) for lw in args.loss_weights]) +  '_tiered_pipeline_lc'
        
        if args.train_spans:
            model_param_str += 'spans'
        if len(model.ablation) > 0:
            model_param_str += '_ablate_'
            model_param_str += '_'.join(model.ablation)

        # Set up output directory
        output_dir = os.path.join(args.output_dir, 'saved_models', model_param_str)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save validation results
        # save_results(metr_attr, output_dir, dev_dataset_name % 'attributes')
        # save_results(metr_prec, output_dir, dev_dataset_name % 'preconditions')
        # save_results(metr_eff, output_dir, dev_dataset_name % 'effects')
        # save_results(metr_conflicts, output_dir, dev_dataset_name % 'conflicts')
        # save_results(metr_stories, output_dir, dev_dataset_name % 'stories')
        # save_results(explanations, output_dir, dev_dataset_name % 'explanations')

        # Just save story preds
        save_preds(dev_ids, all_stories, all_pred_stories, output_dir, dev_dataset_name % 'stories')

        emb = emb.module if hasattr(emb, 'module') else emb
        emb.save_pretrained(output_dir)
        torch.save(model, os.path.join(output_dir, 'classifiers.pth'))
        tokenizer.save_vocabulary(output_dir)

    # Save gamma history (if gamma weighted update)
    if args.objective == 'gamma':
        with open(os.path.join(output_dir, 'gamma_history.pkl'), 'wb') as f:
            pickle.dump(model.gamma_history, f)

    # Test model (#TODO: implement testing)
    print("Testing model")
    metr_attr, all_pred_atts, all_atts, \
    metr_prec, all_pred_prec, all_prec, \
    metr_eff, all_pred_eff, all_eff, \
    metr_conflicts, all_pred_conflicts, all_conflicts, \
    metr_stories, all_pred_stories, all_stories, explanations = evaluate_tiered(
        model,
        test_dataloader,
        device,
        [(accuracy_score, 'accuracy'), (f1_score, 'f1')],
        epoch,
        seg_mode=False,
        return_explanations=True
    )
    explanations = add_entity_attribute_labels(explanations, tiered_dataset['test'], list(att_to_num_classes.keys()))

    test_dataset_name = args.subtask + '_%s_test'
    # save_results(metr_attr, output_dir, test_dataset_name % 'attributes')
    # save_results(metr_prec, output_dir, test_dataset_name % 'preconditions')
    # save_results(metr_eff, output_dir, test_dataset_name % 'effects')
    # save_results(metr_conflicts, output_dir, test_dataset_name % 'conflicts')
    # save_results(metr_stories, output_dir, test_dataset_name % 'stories')
    # save_results(explanations, output_dir, test_dataset_name % 'explanations')

    print('Stories:')
    print_dict(metr_stories)
    print('Conflicts:')
    print_dict(metr_conflicts)
    print('Preconditions:')
    print_dict(metr_prec)
    print('Effects:')
    print_dict(metr_eff)


    consistent_preds = 0
    verifiable_preds = 0
    total = 0
    for expl in explanations:
      if expl['valid_explanation']:
        verifiable_preds += 1
      if expl['story_pred'] == expl['story_label']:
        if len(expl['conflict_pred']) == len(expl['conflict_label']) and expl['conflict_pred'][0] == expl['conflict_label'][0] and expl['conflict_pred'][1] == expl['conflict_label'][1]:
          expl['consistent'] = True
          consistent_preds += 1
        else:
          expl['consistent'] = False
      total += 1

    print('Found %s consistent preds (versus %s verifiable) out of %s total' % (str(consistent_preds), str(verifiable_preds), str(total)))
    print('Consistency: %s' % str(float(consistent_preds) / total))
    


if __name__ == "__main__":
    parser = ArgumentParser(description="Train and test model on TRIP.")
    transformers.logging.set_verbosity_error()

    # Model
    parser.add_argument("--dataset", type=str, default="trip")
    parser.add_argument("--model", type=str, default="roberta")
    parser.add_argument("--ablation", type=list, default=["attributes", "states-logits"])
    parser.add_argument("--subtask", type=str, default="cloze", choices=["cloze", "order"])
    parser.add_argument("--train_spans", type=bool, default=False)
    
    # Objective-related hyperparameters
    parser.add_argument("--objective", type=str, choices=["default", "sigmoid", "gamma"], default="default")
    parser.add_argument("--grad-surgery", type=bool, default=False)
    parser.add_argument("--loss_weights", type=list, default=[0.0, 0.4, 0.4, 0.2, 0.0])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--lambda_const", type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--p_th", type=float, nargs=4, default=[0.0, 0.0, 2.0, 5.0])

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # Logging
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--generate_learning_curve", type=bool, default=True)

    args = parser.parse_args()

    main(args)