from torch.optim.swa_utils import AveragedModel, SWALR


def get_swa(optimizer, model, swa_lr=0.005, anneal_epochs=10, anneal_strategy="cos"):
    '''
    SWALR Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer
        swa_lr (float or list): the learning rate value for all param groups
            together or separately for each group.
        anneal_epochs (int): number of epochs in the annealing phase 
            (default: 10)
        anneal_strategy (str): "cos" or "linear"; specifies the annealing 
            strategy: "cos" for cosine annealing, "linear" for linear annealing
            (default: "cos")
        last_epoch (int): the index of the last epoch (default: 'cos')
    
    '''
    swa_model = AveragedModel(model)
    # swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    # swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=swa_lr)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=anneal_epochs, anneal_strategy=anneal_strategy)

    return swa_scheduler, swa_model
