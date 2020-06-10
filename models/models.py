def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        # assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_attn_gan':
        from .cycle_attn_gan_model import CycleAttnGANModel
        model = CycleAttnGANModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
