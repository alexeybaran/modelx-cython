import math

def power(base, exponent):
    if base < 0 and exponent != math.floor(exponent): raise ValueError('real domain')
    if base == 0 and exponent < 0: raise ZeroDivisionError('negative exponent')
    result=math.pow(base,exponent)
    if not math.isfinite(result): raise OverflowError('non-finite')
    return result
