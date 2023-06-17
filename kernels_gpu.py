import cupy as cp

@cp.fuse(kernel_name="nl_prop")
def nl_prop(A: cp.ndarray, dz: float, alpha: float, V: cp.ndarray, g: float, Isat: float) -> None:
    """A fused kernel to apply real space terms

    Args:
        A (cp.ndarray): The field to propagate
        dz (float): Propagation step in m
        alpha (float): Losses
        V (cp.ndarray): Potential
        g (float): Interactions
        Isat (float): Saturation 
    """
    A_sq = cp.abs(A)**2
    A *= cp.exp(dz*(-alpha/(2*(1+A_sq/Isat)) + 1j * V + 1j*g *
                A_sq/(1+A_sq/Isat)))

@cp.fuse(kernel_name="nl_prop_without_V")
def nl_prop_without_V(A: cp.ndarray, dz: float, alpha: float, g: float,
                        Isat: float) -> None:
    """A fused kernel to apply real space terms

    Args:
        A (cp.ndarray): The field to propagate
        dz (float): Propagation step in m
        alpha (float): Losses
        g (float): Interactions
        Isat (float): Saturation 
    """
    A_sq = cp.abs(A)**2
    A *= cp.exp(dz*(-alpha/(2*(1+A_sq/Isat)) + 1j*g *
                A_sq/(1+A_sq/Isat)))
    
@cp.fuse(kernel_name="nl_prop_c")
def nl_prop_c(A1: cp.ndarray, A2: cp.ndarray, dz: float, alpha: float,
                V: cp.ndarray, g11: float, g12: float,
                Isat: float) -> None:
    """A fused kernel to apply real space terms
    Args:
        A1 (cp.ndarray): The field to propagate (1st component)
        A2 (cp.ndarray): 2nd component
        dz (float): Propagation step in m
        alpha (float): Losses
        V (cp.ndarray): Potential
        g11 (float): Intra-component interactions
        g12 (float): Inter-component interactions
        Isat (float): Saturation parameter of intra-component interaction
    """
    A_sq_1 = cp.abs(A1)**2
    A_sq_2 = cp.abs(A2)**2
    # Losses
    A1 *= cp.exp(dz*(-alpha/(2*(1+A_sq_1/Isat))))
    # Potential
    A1 *= cp.exp(dz*(1j * V))
    # Interactions
    A1 *= cp.exp(dz*(1j*(g11*A_sq_1/(1+A_sq_1/Isat) + g12*A_sq_2)))
                

@cp.fuse(kernel_name="nl_prop_without_V_c")
def nl_prop_without_V_c(A1: cp.ndarray, A2: cp.ndarray, dz: float, alpha: float,
                        g11: float, g12: float,
                        Isat: float) -> None:
    """A fused kernel to apply real space terms
    Args:
        A1 (cp.ndarray): The field to propagate (1st component)
        A2 (cp.ndarray): 2nd component
        dz (float): Propagation step in m
        alpha (float): Losses
        g11 (float): Intra-component interactions
        g12 (float): Inter-component interactions
        Isat (float): Saturation parameter of intra-component interaction
    """
    
    A_sq_1 = cp.abs(A1)**2
    A_sq_2 = cp.abs(A2)**2
    # Losses
    A1 *= cp.exp(dz*(-alpha/(2*(1+A_sq_1/Isat))))
    # Interactions
    A1 *= cp.exp(dz*(1j*(g11*A_sq_1/(1+A_sq_1/Isat) + g12*A_sq_2)))

@cp.fuse(kernel_name='vortex_cp')
def vortex_cp(im: cp.ndarray, i: int, j: int, ii: cp.ndarray, jj: cp.ndarray,
                ll: int) -> None:
    """Generates a vortex of charge l at a position (i,j) on the image im.

    Args:
        im (np.ndarray): Image
        i (int): position row of the vortex
        j (int): position column of the vortex
        ii (int): meshgrid position row (coordinates of the image)
        jj (int): meshgrid position column (coordinates of the image)
        l (int): vortex charge

    Returns:
        None
    """
    im += cp.angle(((ii-i)+1j*(jj-j))**ll)