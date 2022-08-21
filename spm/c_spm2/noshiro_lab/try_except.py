try:
    a="あいうえお"
    raise KeyError
except KeyError:
    print(a)