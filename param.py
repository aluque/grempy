""" This is a library to parse, verify and use parameters mostly
designed for numerical codes. """
import sys
import os
import socket
import warnings
import time

class ParameterNotAllowed(ValueError):
    pass


class Parameter(object):
    """ This is a class containing the description, conversion/verification 
    routines. """
    def __init__(self, name, func=None, doc='', verifiers=None, default=None):
        self.name = name
        self.func = func
        self.doc = doc
        if verifiers is None:
            verifiers = []

        self.verifiers = verifiers
        self.default = default


    def __call__(self, s):
        return self.func(s)


    def __set__(self, obj, value):
        for vf in self.verifiers:
            vf(value)
        obj.values[self.name] = value


    def __get__(self, obj, objtype):
        try:
            return obj.values[self.name]
        except KeyError:
            return self.default


def isspecial(name):
    """  Determines if a parameter name is 'special' and has to be handled
    differently.  Special parameters start and end with an underscore
    (maybe two or more, but this is disencouraged). """
    return (name[0] == '_' and name[-1] == '_')


class ParamContainer(object):
    """ You must sub-class this to implement your own verifiers. """
    def __init__(self):
        self.values = {}

        self.params = {k: v for k, v in type(self).__dict__.iteritems() 
                       if isinstance(v, Parameter)}
        self.param_names = {k for k, v in self.params.iteritems()}
        
        

    def dict_load(self, d, warn_undef=True):
        """ Load the parameters from a dictionary-like object.  All other
        loaders must be based on this one. """
        for key, value in d.iteritems():
            if key in self.param_names:
                try:
                    p = self.params[key]
                    if p.func is not None:
                        value = p.func(value)

                    setattr(self, key, value)
                except Exception as e:
                    raise type(e)('Setting "{}" to {}: {}'
                                  .format(key, str(value), e.message))
            else:
                if not isspecial(key) and warn_undef:
                    warnings.warn("Ignoring undefined parameter '{}'"
                                  .format(key))

    
    

    def asdict(self):
        return {name: getattr(self, name) 
                  for name, p in self.params.iteritems()}

    def metadict(self):
        """ Returns a dictionary extended with metadata.  This is
        the default for dumping. """
        d = {'_timestamp_': time.time(),
             '_ctime_': time.ctime(),
             '_command_': ' '.join(sys.argv),
             '_cwd_': os.getcwd(),
             '_user_': os.getlogin(),
             '_host_': socket.gethostname()}

        d.update(self.asdict())
        return d


    ####
    # Now we implement the loaders / dumpers for the supported formats.
    
    def file_load(self, fname):
        """ Loads a file, deciding its format from the extension. """
        loaders = {'.yaml': self.yaml_loadf,
                   '.json': self.json_loadf,
                   '.h5': self.h5_loadf}

        loader = loaders[os.path.splitext(fname)[1]]
        loader(fname)


    def file_dump(self, fname):
        """ Dumps into a file, deciding its format from the extension. """
        dumpers = {'.yaml': self.yaml_dumpf,
                   '.json': self.json_dumpf,
                   '.h5': self.h5_dumpf}

        dumper = dumpers[os.path.splitext(fname)[1]]
        dumper(fname)

    
    ### YAML loaders / dumpers

    def yaml_loadf(self, fname):
        with open(fname) as fp:
            self.yaml_load(fp)


    def yaml_load(self, fp):
        import yaml

        d = yaml.load(fp)
        self.dict_load(d)


    def yaml_dumpf(self, fname):
        with open(fname, 'w') as fp:
            self.yaml_dump(fp)


    def yaml_dump(self, fp):
        import yaml
        
        yaml.dump(self.metadict(), fp, width=50, indent=4, 
                  default_flow_style=False)


    def yaml_dumps(self):
        import yaml

        yaml.dump(self.metadict())



    ### JSON loaders / dumpers

    def json_loadf(self, fname):
        with open(fname) as fp:
            self.json_load(fp)


    def json_load(self, fp):
        """ Load parameters from a json file. """
        import json

        raw = json.load(fp)
        self.dict_load(raw)


    def json_loads(self, s):
        """ Load parameters from a json-formatted string. """
        import json

        raw = json.loads(s)
        self.dict_load(raw)


    def json_dumpf(self, fname):
        with open(fname, 'w') as fp:
            self.json_dump(fp)


    def json_dump(self, fp):
        import json

        return json.dump(self.metadict(), fp, indent=4)


    def json_dumps(self):
        import json

        return json.dumps(self.metadict())


    ### HDF5 loaders/dumpers
    def h5_loadf(self, fname):
        import h5py

        fp = h5py.File(fname, 'r')
        self.h5_load(fp)
        fp.close()


    def h5_load(self, group):
        """ Load the parameters from an hdf5 file. """
        self.dict_load(group.attrs)


    def h5_dumpf(self, fname):
        import h5py

        fp = h5py.File(fname)
        self.h5_dump(fp)
        fp.close()


    def h5_dump(self, group):
        for k, v in self.metadict().iteritems():
            group.attrs[k] = v


    def rst_dump(self):
        """ Converts the input parameter descriptors into an .rst doc file. """
        def bracket(s):
            return '[%s]' % str(s) if s else ''

        return "\n\n\n".join("``%s``\n\n   %s %s" 
                             % (p.name, p.doc, bracket(p.default)) 
                             for k, p in self.params.iteritems())
            

def param(*args, **kwargs):
    default = kwargs.get('default', '')
    def deco(func):
        return Parameter(func.func_name, func, doc=func.__doc__, 
                         verifiers=args, default=default)
    return deco

                     
# Here we define some decorators for the most common parameter values.
def positive(s):
    if s <= 0:
        raise ParameterNotAllowed("Parameter must be positive")
    return s


def nonnegative(s):
    """ Use this decorator to check that a parameter is not negative. """
    if s <= 0:
        raise ParameterNotAllowed("Parameter must be positive")
    return s



def contained_in(l):
    """ Use this decorator to check that a parameter is contained in 
    a given set/list. """    
    def _contained(s):
        if s not in l:
            raise ParameterNotAllowed("Parameter must be one of {{{}}}"
                                          .format(','.join(str(s) for s in l)))
        return s
    return _contained
