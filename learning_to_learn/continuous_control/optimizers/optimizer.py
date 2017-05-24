class VariableOptimizer(object):
    def __init__(self, name):
        self.name = name

    def get_updates(self, loss, params, opt_params, opt_weights):
        """
        Return tuple (param_updates, opt_weight_updates)
        :param loss: 
        :param params: 
        :param opt_params: 
        :param opt_weights: 
        :return: 
        """
        raise NotImplementedError()

    def get_opt_weights_initial(self, srng, params):
        raise NotImplementedError()

    def get_opt_params_initial(self):
        raise NotImplementedError()
