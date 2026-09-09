import pyomo.environ as pyo

from compas_cra.equilibrium.pyomo_helper import objectives


def objective_coefficients(force_count, penalty):
    weights = (7, 2, 5, 3)
    objective = objectives("rbe", weights, penalty=penalty)
    coefficients = []

    for active_index in range(force_count):
        model = pyo.ConcreteModel()
        model.f_id = pyo.Set(initialize=range(force_count))
        model.f = pyo.Var(model.f_id, initialize=0)
        model.f[active_index].set_value(1)
        coefficients.append(pyo.value(objective(model)))

    return coefficients


def test_rbe_objective_weights_three_component_force_layout():
    assert objective_coefficients(6, penalty=False) == [2, 3, 3, 2, 3, 3]


def test_rbe_objective_weights_four_component_force_layout():
    assert objective_coefficients(8, penalty=True) == [2, 5, 3, 3, 2, 5, 3, 3]


def test_cra_penalty_objective_uses_four_component_force_layout():
    weights = (7, 2, 5, 3)
    model = pyo.ConcreteModel()
    model.f_id = pyo.Set(initialize=range(8))
    model.f = pyo.Var(model.f_id, initialize=1)
    model.alpha = pyo.Var(range(2), initialize={0: 2, 1: 3})

    objective = objectives("cra_penalty", weights)

    assert pyo.value(objective(model)) == 117
