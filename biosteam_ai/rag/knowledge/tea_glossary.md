# Minimum ethanol selling price (MESP)

The minimum ethanol selling price (MESP), sometimes called the minimum selling
price (MSP), is the price at which ethanol must be sold for the project to break
even at the target internal rate of return. It is the headline output of an
ethanol biorefinery TEA. A lower MESP means more competitive economics. In this
tool MESP is reported in both USD per gallon and USD per kilogram.

# Internal rate of return (IRR)

The internal rate of return (IRR) is the discount rate at which the net present
value of the project's cash flows equals zero. TEA typically fixes a target IRR
(for example 10%) and solves for the product price that achieves it. A higher
required IRR raises the minimum selling price, because investors demand a larger
return.

# Total capital investment (TCI)

Total capital investment (TCI) is the total up-front cost to build the plant,
including purchased equipment, installation, and indirect costs. It is derived
from the sizes of the unit operations determined during simulation.

# Operating costs: VOC and FOC

Variable operating cost (VOC) scales with production and includes feedstock,
other raw materials, and utilities. Fixed operating cost (FOC) does not scale
with production and includes labor, maintenance, and overhead. Material cost is
the annual cost of feedstock and other input chemicals.

# Discounted cash flow analysis

TEA evaluates a project over its operating lifetime using discounted cash flow
analysis: annual revenues and costs are projected and discounted to present
value. Key assumptions include the plant lifetime (years), operating days per
year, income tax rate, and the target return.

# Feedstock price

Feedstock price is the purchase cost per unit mass of the raw biomass (corn
stover, sugarcane, etc.). Feedstock is usually one of the largest contributors
to operating cost, so the minimum selling price is highly sensitive to it.

# Life-cycle assessment (LCA)

Life-cycle assessment (LCA) quantifies environmental impacts such as global
warming potential (GWP, or carbon intensity) across a product's life cycle.
BioSTEAM can support LCA when characterization factors are defined for the
streams and utilities. The corn stover and sugarcane models in this tool do not
currently have characterization factors defined, so carbon-intensity results are
not yet available here.
