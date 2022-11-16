from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def get_dataloaders(args, tiered_tensor_dataset):
    train_sampler = RandomSampler(tiered_tensor_dataset['train'])
    train_dataloader = DataLoader(
        tiered_tensor_dataset['train'],
        sampler=train_sampler,
        batch_size=args.batch_size
    )

    dev_sampler = SequentialSampler(tiered_tensor_dataset['dev'])
    dev_dataloader = DataLoader(
        tiered_tensor_dataset['dev'],
        sampler=dev_sampler,
        batch_size=args.eval_batch_size
    )

    test_sampler = SequentialSampler(tiered_tensor_dataset['test'])
    test_dataloader = DataLoader(
        tiered_tensor_dataset['test'],
        sampler=test_sampler,
        batch_size=args.test_batch_size
    )

    return train_dataloader, dev_dataloader, test_dataloader