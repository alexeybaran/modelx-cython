"""Completed-cache serialization shared by original and prepared owners.

Only completed entries are published; formula execution and cache layout are
unchanged. This is the original owner implementation moved to one emitter.
"""


def methods(manifest):
    lines = []
    typed = {n: r for n, r in manifest['cells'].items()
             if r.get('storage') in {'typed_scalar', 'typed_epoch_slabs', 'typed_python_dict'}}
    lines += ['    def snapshot(self):', '        result = []', '        cdef Py_ssize_t i']
    objects = set(manifest.get('object_cells', ()))
    growing = {n: r for n, r in manifest['cells'].items() if r.get('storage') == 'growing_numeric'}
    numeric_rows = {n: r for n, r in manifest['cells'].items()
                    if n not in objects and n not in growing and n not in typed and r['cached']}
    for name in growing:
        lines += [f'        result.append((({name!r}, "numeric"), self._numeric_snapshot_{name}()))']
    for name in typed:
        lines += [f'        result.append((({name!r}, "typed"), self._typed_snapshot_{name}()))']
    for name, row in numeric_rows.items():
        value = '_floats' if row['result']['kind'] == 'float' else '_ints'
        lines += [f"        for i in range({row['offset']}, {row['offset'] + row['slots']}):",
                  '            if self._frame._seen[i] == self._frame._epoch:',
                  f'                result.append((i, self._frame.{value}[i]))']
    for name in sorted(objects):
        row = manifest['cells'][name]
        if not row['cached'] or name in typed:
            continue
        if row['storage'] == 'object_singleton':
            lines += [f'        if self._object_initialized and self._object_singleton_{name} is not _DICT_EMPTY and self._object_singleton_{name} is not _DICT_BUSY:',
                      f'            result.append((({name!r}, "object"), self._object_singleton_{name}))']
        elif row['storage'] == 'python_dict':
            lines += ['        if self._object_initialized:',
                      f'            for key, value in self._object_{name}.items():',
                      '                if value is not _DICT_EMPTY and value is not _DICT_BUSY:',
                      f'                    result.append((({name!r}, key), value))']

        elif row['storage'] == 'object_slots':
            lines += ['        if self._object_initialized:',
                      f'            for i in range({row["slots"]}):',
                      f'                entry = self._object_{name}[i]',
                      '                if entry is not None and (<_ObjectEntry>entry).state == 2:',
                      f'                    result.append((({name!r}, i), (<_ObjectEntry>entry).value))']
    lines += ['        return result', '    def restore(self, entries):', '        cdef Py_ssize_t i']
    if objects:
        lines.append('        cdef _ObjectEntry entry')
    lines.append('        for key, value in entries:')
    if objects or growing:
        lines += ['            if isinstance(key, tuple):',
                  '                name, index = key']
        cases = 0
        for name in growing:
            lines += [f'                {"if" if cases == 0 else "elif"} name == {name!r}:',
                      '                    if index != "numeric": raise ValueError("Invalid numeric cache snapshot")',
                      f'                    self._numeric_restore_{name}(value)']
            cases += 1
        for name in typed:
            lines += [f'                {"if" if cases == 0 else "elif"} name == {name!r}:',
                      '                    if index != "typed": raise ValueError("Invalid typed cache snapshot")',
                      f'                    self._typed_restore_{name}(value)']
            cases += 1
        for name in sorted(objects):
            row = manifest['cells'][name]
            if not row['cached'] or name in typed:
                continue
            lines += [f'                {"if" if cases == 0 else "elif"} name == {name!r}:']
            cases += 1
            if row['storage'] == 'object_singleton':
                lines += ['                    if index != "object": raise ValueError("Invalid completed object slot")',
                          f'                    self._object_singleton_{name} = value']
            elif row['storage'] == 'python_dict':
                lines += [f'                    self._object_{name}[index] = value']

            elif row['storage'] == 'object_slots':
                lines += [f'                    if not 0 <= index < {row["slots"]}: raise ValueError("Invalid completed object slot")',
                          '                    entry = _ObjectEntry()',
                          '                    entry.state = 2', '                    entry.value = value',
                          '                    entry.touched = True',
                          '                    if self._object_dirty is None: self._object_dirty = []',
                          '                    self._object_dirty.append(entry)',
                          f'                    self._object_{name}[index] = entry']
        if cases:
            lines += ['                else: raise ValueError("Unknown completed object Cell")']
        else:
            lines += ['                raise ValueError("No cached object Cell")']
        lines += ['                continue']
    lines += ['            i = key']
    for j, row in enumerate(numeric_rows.values()):
        value = '_floats' if row['result']['kind'] == 'float' else '_ints'
        lines += [f"            {'if' if j == 0 else 'elif'} {row['offset']} <= i < {row['offset'] + row['slots']}:",
                  f'                self._frame.{value}[i] = value']
    if numeric_rows:
        lines += ['            else: raise ValueError("Invalid completed cache slot")',
                  '            self._frame._seen[i] = self._frame._epoch']
    else:
        lines.append('            raise ValueError("No numeric cache slots")')
    return lines
