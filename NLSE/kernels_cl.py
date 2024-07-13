from pyopencl import array as cla
from pyopencl import clmath


def nl_prop(
    A: cla.Array,
    A_sq: cla.Array,
    dz: float,
    alpha: float,
    V: cla.Array,
    g: float,
    Isat: float,
) -> None:
    """A fused kernel to apply real space terms

    Args:
        A (cla.Array): The field to propagate
        A_sq (cla.Array): The field modulus squared
        dz (float): Propagation step in m
        alpha (float): Losses
        V (cla.Array): Potential
        g (float): Interactions
        Isat (float): Saturation
    """
    # saturation
    sat = 1 / (1 + A_sq / Isat)
    # Interactions
    arg = 1j * g * A_sq * sat
    # Losses
    arg += -alpha * sat
    # Potential
    arg += 1j * V
    arg *= dz
    arg = clmath.exp(arg)
    A *= arg


def nl_prop_without_V(
    A: cla.Array,
    A_sq: cla.Array,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """A fused kernel to apply real space terms

    Args:
        A (cla.Array): The field to propagate
        A_sq (cla.Array): The field modulus squared
        dz (float): Propagation step in m
        alpha (float): Losses
        g (float): Interactions
        Isat (float): Saturation
    """
    # saturation
    sat = 1 / (1 + A_sq / Isat)
    # Interactions
    arg = 1j * g * A_sq * sat
    # Losses
    arg += -alpha * sat
    arg *= dz
    arg = clmath.exp(arg)
    A *= arg


def nl_prop_c(
    A1: cla.Array,
    A_sq_1: cla.Array,
    A_sq_2: cla.Array,
    dz: float,
    alpha: float,
    V: cla.Array,
    g11: float,
    g12: float,
    Isat: float,
) -> None:
    """A fused kernel to apply real space terms
    Args:
        A1 (cla.Array): The field to propagate (1st component)
        A_sq_1 (cla.Array): The field modulus squared (1st component)
        A_sq_2 (cla.Array): The field modulus squared (2nd component)
        dz (float): Propagation step in m
        alpha (float): Losses
        V (cla.Array): Potential
        g11 (float): Intra-component interactions
        g12 (float): Inter-component interactions
        Isat1 (float): Saturation parameter of first component
        Isat2 (float): Saturation parameter of second component
    """
    # Saturation parameter
    sat = 1 / (1 + (A_sq_1 + A_sq_2) / Isat)
    # Interactions
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat)
    # Losses
    arg += -alpha * sat
    # Potential
    arg += 1j * V
    arg *= dz
    arg = clmath.exp(arg)
    A1 *= arg


def nl_prop_without_V_c(
    A1: cla.Array,
    A_sq_1: cla.Array,
    A_sq_2: cla.Array,
    dz: float,
    alpha: float,
    g11: float,
    g12: float,
    Isat: float,
) -> None:
    """A fused kernel to apply real space terms
    Args:
        A1 (cla.Array): The field to propagate (1st component)
        A_sq_1 (cla.Array): The field modulus squared (1st component)
        A_sq_2 (cla.Array): The field modulus squared (2nd component)
        dz (float): Propagation step in m
        alpha (float): Losses
        g11 (float): Intra-component interactions
        g12 (float): Inter-component interactions
        Isat1 (float): Saturation parameter of first component
        Isat2 (float): Saturation parameter of second component
    """
    # Saturation parameter
    sat = 1 / (1 + A_sq_1 / Isat)
    # Interactions
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat)
    # Losses
    arg += -alpha * sat
    arg *= dz
    arg = clmath.exp(arg)
    A1 *= arg


def rabi_coupling(A, dz: float, omega: float) -> None:
    """Apply a Rabi coupling term.
    This function implements the Rabi hopping term.
    It exchanges density between the two components.

    Args:
        A (cla.Array): First field / component
        dz (float): Solver step
        omega (float): Rabi coupling strength
    """
    A1 = A[..., 0, :, :]
    A2 = A[..., 1, :, :]
    A1_old = A1.copy()
    A1[:] = clmath.cos(omega * dz) * A1 - 1j * clmath.sin(omega * dz) * A2
    A2[:] = clmath.cos(omega * dz) * A2 - 1j * clmath.sin(omega * dz) * A1_old


def vortex_cp(
    im: cla.Array, i: int, j: int, ii: cla.Array, jj: cla.Array, ll: int
) -> None:
    """Generates a vortex of charge l at a position (i,j) on the image im.

    Args:
        im (np.ndarray): Image
        i (int): position row of the vortex
        j (int): position column of the vortex
        ii (int): meshgrid position row (coordinates of the image)
        jj (int): meshgrid position column (coordinates of the image)
        ll (int): vortex charge

    Returns:
        None
    """
    im += clmath.atan(((ii - i) + 1j * (jj - j)) ** ll)


def square_mod(A: cla.Array, A_sq: cla.Array) -> None:
    """Compute the square modulus of the field

    Args:
        A (cla.Array): The field
        A_sq (cla.Array): The modulus squared of the field

    Returns:
        None
    """
    A_sq[:] = A.real * A.real + A.imag * A.imag
