"""Header-only signed-int64 range cursor; no runtime module dependency."""

cdef struct I64Range:
    long long current
    long long stop
    long long step
    bint active


cdef inline void range_init(I64Range* state, long long start, long long stop, long long step) except *:
    if step == 0:
        raise ValueError('range() arg 3 must not be zero')
    state.current, state.stop, state.step = start, stop, step
    state.active = start < stop if step > 0 else start > stop


cdef inline bint range_next(I64Range* state, long long* value) noexcept:
    cdef unsigned long long distance, magnitude
    if not state.active:
        return False
    value[0] = state.current
    if state.step > 0:
        distance = <unsigned long long>state.stop - <unsigned long long>state.current
        magnitude = <unsigned long long>state.step
    else:
        distance = <unsigned long long>state.current - <unsigned long long>state.stop
        magnitude = <unsigned long long>0 - <unsigned long long>state.step
    if distance <= magnitude:
        state.active = False
    else:
        # The next value is strictly inside signed-int64 endpoints. Do not
        # calculate the potentially overflowing terminating increment.
        state.current += state.step
    return True
