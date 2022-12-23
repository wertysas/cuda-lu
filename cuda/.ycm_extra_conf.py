def Settings( **kwargs ):
  return {
    'flags': [ '-x', 'cuda', '-nocudainc', '-nocudalib', '-std=c++11', '-Wall', '-Wextra', '-Werror' ],
  }
