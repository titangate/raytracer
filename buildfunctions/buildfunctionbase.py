class BuildFunctionBase(object):
    @classmethod
    def get_build_function(cls, name):
        for subclass in BuildFunctionBase.__subclasses__():
            if subclass.BUILD_FUNCTION_NAME == name:
                return subclass.build_function

        return None
