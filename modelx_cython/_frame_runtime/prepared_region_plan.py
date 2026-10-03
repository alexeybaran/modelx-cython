"""Resolved resource specifications for the original package compilation plan.

This supplies reference context to the existing Cython emitter. Original
discovery, tracing, usage analysis and signatures remain authoritative. No
formula execution, alternate frontend or package facade is introduced here.
"""
from dataclasses import replace
from pathlib import Path

from .compilation_plan_adapter import from_compilation_plan


OPTIONS = frozenset(('point_table', 'point_parameter', 'steps', 'row_references',
                     'parameter_initial_values', 'object_bindings',
                     'typed_mapping_inputs', 'typed_object_inputs',
                     'native_typed_inputs'))


def resolve(package_plan, exported, destination):
    """Resolve optional rich input declarations after original type finalization.

    records_file contains reference examples used only while compiling dynamic
    row bindings. At runtime the original Space formula supplies those rows.
    Ordinary models need neither this declaration nor an artificial policy table.
    """
    destination = Path(destination)
    contexts = {}
    for identity, space in package_plan.spaces.items():
        if space.native_context is None:
            continue
        options = dict(space.native_context)
        extra = options.keys() - OPTIONS - {'records_file'}
        if extra:
            raise ValueError('Unknown native resource options for ' + identity + ': ' + repr(sorted(extra)))
        if space.locked:
            raise ValueError('Locked Space retains its synchronized generation: ' + identity)
        filename = options.pop('records_file', None)
        if filename is not None:
            import pandas as pd
            path = Path(filename)
            if not path.is_absolute():
                path = destination.parent / path
            options['policy_records'] = pd.read_pickle(path)
        # Types are already fixed by the original trace/spec/usage stages.
        # Object-capable formulas stay compiled within this same representation.
        options['cython_objects'] = True
        projection = identity.removeprefix(package_plan.model + '.')
        spec = from_compilation_plan(package_plan, exported,
                                    destination / ('_mx_context_source_' + space.class_name),
                                    projection=projection, **options)
        contexts[space.class_identity] = replace(spec, public_emission_plan=space)
    return contexts
