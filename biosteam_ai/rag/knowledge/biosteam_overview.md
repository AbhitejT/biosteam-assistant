# What BioSTEAM is

BioSTEAM (the Biorefinery Simulation and Techno-Economic Analysis Modules) is an
open-source, steady-state process simulator written in Python. It streamlines
the design, simulation, techno-economic analysis (TEA), and life-cycle
assessment (LCA) of chemical processes, with a focus on biorefineries. It is
built to evaluate many design scenarios quickly and to support rigorous
sensitivity and uncertainty analysis.

# How a BioSTEAM simulation works

A biorefinery is represented as a flowsheet of connected unit operations (for
example mixers, reactors, distillation columns, heat exchangers, and pumps).
Material streams flow between units. When the system is simulated, BioSTEAM
solves the mass and energy balances for every unit, sizes the equipment, and
estimates utility (heating, cooling, electricity) requirements. Because
biorefineries contain recycle loops, BioSTEAM iterates until the recycle streams
converge to a consistent solution. The solver warm-starts from the previous
solution, so repeated simulations after small parameter changes are fast.

# Techno-economic analysis (TEA) in BioSTEAM

Once the process is simulated, a TEA object computes the economics: capital
costs from equipment sizing, operating costs from materials and utilities, and a
discounted cash flow over the plant lifetime. A common use is to solve for the
product selling price that achieves a target internal rate of return; for an
ethanol biorefinery this is the minimum ethanol selling price (MESP).

# Uncertainty and sensitivity

BioSTEAM is designed to handle uncertainty. Instead of a single point estimate,
you can vary uncertain inputs (such as feedstock price or fermentation
conversion) across ranges or probability distributions and observe how outputs
such as MESP respond. Sensitivity analysis sweeps one parameter; uncertainty
(Monte Carlo) analysis samples many parameters from distributions to produce a
range of outcomes with percentiles.

# The biorefinery model library

Complete biorefinery configurations live in the Bioindustrial-Park repository,
BioSTEAM's library of biorefinery models and results. These include
conventional and cellulosic crops, municipal solid waste, organic acids,
oleochemicals, and biofuels, plus newer thermochemical routes such as
waste-plastics pyrolysis.
