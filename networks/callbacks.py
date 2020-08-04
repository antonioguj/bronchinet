

class CallbackBase(object):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError