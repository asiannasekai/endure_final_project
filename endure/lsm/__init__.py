from .cost import Cost
from .data_generator import (
        LSMDataGenerator,
        ClassicGen,
        TieringGen,
        LevelingGen,
        QHybridGen,
        FluidLSMGen,
        KapacityGen
    )
from .types import Policy, System, LSMDesign, LSMBounds, Workload


def build_data_gen(policy: Policy, bounds: LSMBounds, **kwargs) -> LSMDataGenerator:
    generators = {
        Policy.Classic: ClassicGen,
        Policy.Tiering: TieringGen,
        Policy.Leveling: LevelingGen,
        Policy.QHybrid: QHybridGen,
        Policy.Fluid: FluidLSMGen,
        Policy.Kapacity: KapacityGen,
    }
    generator_class = generators.get(policy, None)
    if generator_class is None:
        raise KeyError
    generator = generator_class(bounds, **kwargs)

    return generator
