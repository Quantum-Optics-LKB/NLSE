import cupy as cp


@cp.fuse(kernel_name="nl_prop")
def nl_prop(
    A: cp.ndarray,
    A_sq: cp.ndarray,
    dz: float,
    alpha: float,
    V: cp.ndarray,
    g: float,
    Isat: float,
) -> None:
    """A fused kernel to apply real space terms

    Args:
        A (cp.ndarray): The field to propagate
        A_sq (cp.ndarray): The field modulus squared
        dz (float): Propagation step in m
        alpha (float): Losses
        V (cp.ndarray): Potential
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
    cp.exp(arg, out=arg)
    A *= arg


@cp.fuse(kernel_name="nl_prop_without_V")
def nl_prop_without_V(
    A: cp.ndarray,
    A_sq: cp.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """A fused kernel to apply real space terms

    Args:
        A (cp.ndarray): The field to propagate
        A_sq (cp.ndarray): The field modulus squared
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
    cp.exp(arg, out=arg)
    A *= arg


@cp.fuse(kernel_name="nl_prop_c")
def nl_prop_c(
    A1: cp.ndarray,
    A_sq_1: cp.ndarray,
    A_sq_2: cp.ndarray,
    dz: float,
    alpha: float,
    V: cp.ndarray,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> None:
    """A fused kernel to apply real space terms
    Args:
        A1 (cp.ndarray): The field to propagate (1st component)
        A_sq_1 (cp.ndarray): The field modulus squared (1st component)
        A_sq_2 (cp.ndarray): The field modulus squared (2nd component)
        dz (float): Propagation step in m
        alpha (float): Losses
        V (cp.ndarray): Potential
        g11 (float): Intra-component interactions
        g12 (float): Inter-component interactions
        Isat1 (float): Saturation parameter of first component
        Isat2 (float): Saturation parameter of second component
    """
    # Saturation parameter
    sat = 1 / (1 + A_sq_1 * 1 / Isat1 + A_sq_2 * 1 / Isat2)
    # Interactions
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat)
    # Losses
    arg += -alpha * sat
    # Potential
    arg += 1j * V
    arg *= dz
    cp.exp(arg, out=arg)
    A1 *= arg


@cp.fuse(kernel_name="nl_prop_without_V_c")
def nl_prop_without_V_c(
    A1: cp.ndarray,
    A_sq_1: cp.ndarray,
    A_sq_2: cp.ndarray,
    dz: float,
    alpha: float,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> None:
    """A fused kernel to apply real space terms
    Args:
        A1 (cp.ndarray): The field to propagate (1st component)
        A_sq_1 (cp.ndarray): The field modulus squared (1st component)
        A_sq_2 (cp.ndarray): The field modulus squared (2nd component)
        dz (float): Propagation step in m
        alpha (float): Losses
        g11 (float): Intra-component interactions
        g12 (float): Inter-component interactions
        Isat1 (float): Saturation parameter of first component
        Isat2 (float): Saturation parameter of second component
    """
    # Saturation parameter
    sat = 1 / (1 + A_sq_1 * 1 / Isat1 + A_sq_2 * 1 / Isat2)
    # Interactions
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat)
    # Losses
    arg += -alpha * sat
    arg *= dz
    cp.exp(arg, out=arg)
    A1 *= arg


@cp.fuse(kernel_name="rabi_coupling")
def rabi_coupling(A1: cp.array, A2: cp.array, dz: float, omega: float) -> None:
    """Apply a Rabi coupling term.
    This function implements the Rabi hopping term.
    It exchanges density between the two components.

    Args:
        A1 (cp.ndarray): First field / component
        A2 (cp.ndarray): Second field / component
        dz (float): Solver step
        omega (float): Rabi coupling strength
    """
    A1_old = A1.copy()
    A1[:] = cp.cos(omega * dz) * A1 - 1j * cp.sin(omega * dz) * A2
    A2[:] = cp.cos(omega * dz) * A2 - 1j * cp.sin(omega * dz) * A1_old


@cp.fuse(kernel_name="vortex_cp")
def vortex_cp(
    im: cp.ndarray, i: int, j: int, ii: cp.ndarray, jj: cp.ndarray, ll: int
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
    im += cp.angle(((ii - i) + 1j * (jj - j)) ** ll)


@cp.fuse(kernel_name="square_mod_cp")
def square_mod(A: cp.ndarray, A_sq: cp.ndarray) -> None:
    """Compute the square modulus of the field

    Args:
        A (cp.ndarray): The field
        A_sq (cp.ndarray): The modulus squared of the field

    Returns:
        None
    """
    A_sq[:] = cp.abs(A) ** 2
