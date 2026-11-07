"""Use conditional error checks on private numeric functions we emit."""
import libcst as cst


class NumericExceptions(cst.CSTTransformer):
    def visit_ClassDef(self, node):
        return False

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
