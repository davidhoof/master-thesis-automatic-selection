class LayerHook:
    """
    Utility class to add a forward hook on a torch-CNN-Layer
    """

    def __init__(self):
        self.storage = None
        self.hook_handle = None

    def pull(self):
        """
        Get the storage of the hook
        :return: storage
        """
        if self.hook_handle is not None:
            self.hook_handle.remove()
        return self.storage

    def register_hook(self, module, store_input=True):
        """
        Register a hook on a module
        :param module: module on which the forward hook is attached
        :param store_input: Toggle if the input or output of the module is stored
        """
        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.storage = None

        def hook(model, inp, out):
            if store_input:
                self.storage = inp
            else:
                self.storage = out

        self.hook_handle = module.register_forward_hook(hook)
