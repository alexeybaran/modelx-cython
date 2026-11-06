"""Use Cython conditional error checks for numeric generated signatures."""
import libcst as cst


def clause(type_):
    return ' except? -1' if type_ in ('double', 'long long', 'bint') else ''


class NumericExceptions(cst.CSTTransformer):
    def leave_FunctionDef(self, before, after):
        module = cst.Module([])
        decorators = [module.code_for_node(d.decorator) for d in after.decorators]
        if (after.returns and any(d in ('_mx_cy.cfunc', '_mx_cy.ccall') for d in decorators)
                and not any(d.startswith('_mx_cy.exceptval(') for d in decorators)
                and module.code_for_node(after.returns.annotation)
                    in ('_mx_cy.double', '_mx_cy.longlong', '_mx_cy.bint')):
            return after.with_changes(decorators=(*after.decorators,
                cst.Decorator(cst.parse_expression('_mx_cy.exceptval(-1, check=True)'))))
        return after
