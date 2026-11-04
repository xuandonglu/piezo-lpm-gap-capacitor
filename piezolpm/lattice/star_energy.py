"""Energy and energy gradient of the plane piezoelectric lattice particle model (direct particle stars).

Each particle carries a star of eight bonds (four axial, four diagonal) to its neighbours on a square lattice.
The star energy contains pair terms (spring, piezoelectric, dielectric) and the collective terms of the lattice
particle model.  Directions missing at free surfaces and crack faces are completed from same-side neighbours
(affine completion); at a crack tip the star is split into cell sectors that share the tip degrees of freedom.

Nine dimensionless coordinates times parameter_scales give the physical constants
[C11_2D, C22_2D, C12_2D, C66_2D, e15, e31, e33, kappa11, kappa22]; the first four correspond to the 3D
C11/C33/C13/C55, respectively, and kappa is absolute (F/m).  Scope: homogeneous axis-aligned orthotropic
material in plane strain, poled along y, square one-shell lattice, zero-width cracks.  The default constants are
those of the numerical cases.  Energies and gradients are evaluated by index-add operations
on the bond/star incidence; the case scripts extract the sparse matrix from the gradient where a direct solver
is used.
"""

import numpy as np
import torch


PARAMETER_NAMES = ('C11', 'C22', 'C12', 'C66', 'e15', 'e31', 'e33', 'kappa11', 'kappa22')
DEFAULT_SCALES = np.array([1e11]*4+[10.]*3+[1e-8]*2)
EPS0 = 8.8541878128e-12
DEFAULT_PHYSICAL = np.array([139e9,115e9,74.3e9,25.6e9,12.7,-5.2,15.1,
                             EPS0*730.,EPS0*635.])
_OFFSETS = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], float)
_DIRECTIONS = _OFFSETS/np.linalg.norm(_OFFSETS,axis=1)[:,None]


def _integer_array(value, shape_tail, name):
    raw = np.asarray(value)
    if raw.size == 0:
        return np.empty((0,shape_tail),dtype=np.int64)
    if (raw.ndim != 2 or raw.shape[1] != shape_tail
            or not np.issubdtype(raw.dtype,np.integer)):
        raise ValueError(name+' must be a two-dimensional integer index array')
    return raw.astype(np.int64,copy=True)


def _physical_parameters(material):
    if material is None:
        return DEFAULT_PHYSICAL.copy()
    if all(hasattr(material,name) for name in ('c','e','kappa')):
        raw_arrays = [np.asarray(getattr(material,name)) for name in ('c','e','kappa')]
        if any(np.iscomplexobj(a) for a in raw_arrays):
            raise ValueError('This static planar model requires real coefficients')
        c,e,k = (np.asarray(a,float) for a in raw_arrays)
        if c.shape != (3,3) or e.shape != (2,3) or k.shape != (2,2):
            raise ValueError('Expected planar c[3,3], e[2,3], kappa[2,2]')
        if not all(np.all(np.isfinite(a)) for a in (c,e,k)):
            raise ValueError('Nonfinite material coefficient')
        if not np.allclose(c,c.T,rtol=1e-12,atol=0.) or not np.allclose(k,k.T,rtol=1e-12,atol=0.):
            raise ValueError('Material C and kappa must be symmetric')
        if (np.max(abs(c[[0,1],2])) > 1e-12*np.max(abs(c))
                or max(abs(e[0,0]),abs(e[0,1]),abs(e[1,2])) > 1e-12*max(np.max(abs(e)),1.)
                or abs(k[0,1]) > 1e-12*np.max(abs(k))):
            raise ValueError('Only axis-aligned orthotropic C, y-poling and diagonal kappa are supported')
        return np.array([c[0,0],c[1,1],c[0,1],c[2,2],e[0,2],e[1,0],e[1,1],k[0,0],k[1,1]])
    raw = np.asarray(material)
    if np.iscomplexobj(raw):
        raise ValueError('This static planar model requires real coefficients')
    values = np.asarray(material,float)
    if values.shape != (9,) or not np.all(np.isfinite(values)):
        raise ValueError('Expected nine finite physical material coefficients')
    return values.copy()


def _validate_physical(values):
    """Scalar real-material/full-completed-star checks; no eigenproblem."""
    values = np.asarray(values,float)
    if values.shape != (9,) or not np.all(np.isfinite(values)):
        raise ValueError('Expected nine finite real material coefficients')
    c11,c22,c12,c66,_,_,_,k11,k22 = values
    if (min(c11,c22,c66,k11,k22) <= 0
            or abs(c12) >= np.sqrt(c11)*np.sqrt(c22)):
        raise ValueError('Real C and fixed-strain kappa must be positive definite')
    kx,ky,kd,tv = (c11-c12)/2,(c22-c12)/2,c66/2,(c12-c66)/12
    if min(kx,ky,kd) < 0:
        raise ValueError('Completed direct-star mechanical metric is not PSD')
    # Full eight-direction metric: axial diag(kx,kx,ky,ky)+tv*11.T,
    # diagonal kd*I+tv*11.T.  For tv<0 use the exact rank-one criteria.
    if tv < 0 and (min(kx,ky) <= 0 or 1+tv*(2/kx+2/ky) < -1e-12
                   or kd+4*tv < -1e-12*max(kd,abs(tv))):
        raise ValueError('Completed direct-star mechanical metric is not PSD')


def _geometry(mesh,thickness):
    """Return flattened scalar difference incidence, never Hessian/D matrices."""
    xy = np.asarray(mesh.coordinates,float).copy()
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.all(np.isfinite(xy)):
        raise ValueError('Finite planar coordinates required')
    n = len(xy)
    h = float(mesh.spacing)
    volumes = np.asarray(mesh.volumes,float).copy()
    cells = _integer_array(mesh.cells,4,'cells')
    pairs = _integer_array(mesh.pairs,2,'pairs')
    if (not np.isfinite(h) or h <= 0 or volumes.shape != (n,)
            or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0)
            or not len(cells) or np.any(cells < 0) or np.any(cells >= n)
            or np.any(pairs < 0) or np.any(pairs >= n)):
        raise ValueError('Invalid square-cell particle geometry')
    incident = [[] for _ in range(n)]
    cell_areas = np.empty(len(cells))
    quarter_volumes = np.zeros(n)
    expected_pairs = set()
    for ci,conn in enumerate(cells):
        p = xy[conn]
        area = .5*abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))
        q = (p-p.min(axis=0))/h
        corners = set(map(tuple,np.rint(q).astype(int)))
        if (len(set(map(int,conn))) != 4 or area <= 0
                or not np.allclose(q,np.rint(q),rtol=0,atol=1e-8)
                or corners != {(0,0),(1,0),(1,1),(0,1)}):
            raise ValueError('Only actual axis-aligned square geometry cells are supported')
        cell_areas[ci] = area
        quarter_volumes[conn] += area/4
        for a in range(4):
            incident[int(conn[a])].append(ci)
            for b in range(a+1,4):
                expected_pairs.add(tuple(sorted((int(conn[a]),int(conn[b])))))
    if not np.allclose(quarter_volumes,volumes,rtol=1e-12,atol=1e-18):
        raise ValueError('Particle volumes must be the actual quarter-cell areas')
    actual_pairs = {tuple(sorted(map(int,pair))) for pair in pairs}
    if actual_pairs != expected_pairs or len(actual_pairs) != len(pairs):
        raise ValueError('Real pair graph must match actual geometry cells without duplicate bonds')
    neighbors = [[] for _ in range(n)]
    for a,b in pairs:
        neighbors[a].append(int(b)); neighbors[b].append(int(a))

    entry_node,entry_bond,entry_value = [],[],[]
    bond_star,bond_direction,bond_kind = [],[],[]
    star_scale,star_center,star_cell = [],[],[]
    star_node,star_node_star,axial_coeff,diagonal_coeff = [],[],[],[]
    ambiguous_nodes = []
    completed_boundary_stars = virtual_count = 0

    def append_star(center,nodes,lookup,derivative,scale,cell_index):
        nonlocal completed_boundary_stars,virtual_count
        si = len(star_scale)
        star_scale.append(scale); star_center.append(center); star_cell.append(cell_index)
        # Compact star-node incidence, used to aggregate collective diagonal
        # coefficients BEFORE squaring. No local Hessian is constructed.
        ac = np.zeros((len(nodes),2)); dc = np.zeros_like(ac)
        missing = 0
        for row,offset in enumerate(_OFFSETS):
            position = lookup.get(tuple(offset.astype(int)))
            coefficients = np.zeros(len(nodes))
            if position is not None:
                coefficients[position],coefficients[0] = 1.,-1.
            else:
                coefficients = h*(offset[0]*derivative[0]+offset[1]*derivative[1])
                missing += 1
            bi = len(bond_star)
            bond_star.append(si); bond_direction.append(_DIRECTIONS[row])
            bond_kind.append(row)
            for local in np.flatnonzero(coefficients):
                entry_node.append(int(nodes[local])); entry_bond.append(bi)
                entry_value.append(float(coefficients[local]))
            (ac if row < 4 else dc)[:] += coefficients[:,None]*_DIRECTIONS[row]
        for local,node in enumerate(nodes):
            star_node.append(int(node)); star_node_star.append(si)
            axial_coeff.append(ac[local]); diagonal_coeff.append(dc[local])
        completed_boundary_stars += int(missing > 0)
        virtual_count += missing

    for center,nb in enumerate(neighbors):
        nodes = np.array([center]+sorted(nb),dtype=int)
        q = (xy[nodes[1:]]-xy[center])/h
        if (not np.allclose(q,np.rint(q),rtol=0,atol=1e-8)
                or np.any(abs(q)>1+1e-8) or np.any(np.linalg.norm(q,axis=1)<.9)):
            raise ValueError('Direct-star messages require square one-shell real bonds')
        keys = [tuple(v) for v in np.rint(q).astype(int)]
        ambiguous = len(set(keys)) != len(keys)
        if ambiguous:
            ambiguous_nodes.append(center)
            sector_area = 0.
            for ci in incident[center]:
                conn = cells[ci]
                sector_nodes = np.array([center]+sorted(int(i) for i in conn if i != center))
                offsets = np.rint((xy[sector_nodes[1:]]-xy[center])/h).astype(int)
                lookup = {tuple(v):j+1 for j,v in enumerate(offsets)}
                derivative = np.zeros((2,len(sector_nodes)))
                for axis in (0,1):
                    options = [(key,j) for key,j in lookup.items()
                               if abs(key[axis]) == 1 and key[1-axis] == 0]
                    if len(options) != 1:
                        raise ValueError('Each tip sector needs one real axial neighbor per axis')
                    key,j = options[0]
                    derivative[axis,j],derivative[axis,0] = key[axis]/h,-key[axis]/h
                area = cell_areas[ci]/4
                append_star(center,sector_nodes,lookup,derivative,thickness*area/h**2,ci)
                sector_area += area
            if not np.isclose(sector_area,volumes[center],rtol=1e-12,atol=1e-18):
                raise ValueError('Tip sectors do not partition the real particle area')
        else:
            lookup = {key:j+1 for j,key in enumerate(keys)}
            derivative = np.zeros((2,len(nodes)))
            for axis in (0,1):
                positive = tuple(int(v) for v in np.eye(2)[axis])
                negative = tuple(-int(v) for v in np.eye(2)[axis])
                plus,minus = lookup.get(positive),lookup.get(negative)
                if plus is not None and minus is not None:
                    derivative[axis,plus],derivative[axis,minus] = 1/(2*h),-1/(2*h)
                elif plus is not None:
                    derivative[axis,plus],derivative[axis,0] = 1/h,-1/h
                elif minus is not None:
                    derivative[axis,0],derivative[axis,minus] = 1/h,-1/h
                else:
                    raise ValueError('No same-side axial support for virtual directions')
            append_star(center,nodes,lookup,derivative,thickness*volumes[center]/h**2,-1)
    data = dict(coordinates=xy,volumes=volumes,
        coefficient_node=np.array(entry_node,dtype=np.int64),
        coefficient_bond=np.array(entry_bond,dtype=np.int64),
        coefficient_value=np.array(entry_value),bond_star=np.array(bond_star,dtype=np.int64),
        bond_direction=np.array(bond_direction),bond_kind=np.array(bond_kind,dtype=np.int64),
        star_scale=np.array(star_scale),star_center=np.array(star_center,dtype=np.int64),
        star_cell=np.array(star_cell,dtype=np.int64),star_node=np.array(star_node,dtype=np.int64),
        star_node_star=np.array(star_node_star,dtype=np.int64),
        axial_node_coeff=np.array(axial_coeff),diagonal_node_coeff=np.array(diagonal_coeff))
    audit = dict(ambiguous_real_center_nodes=ambiguous_nodes,
        tip_sector_stars=int(np.count_nonzero(data['star_cell'] >= 0)),
        completed_stars_with_virtual_directions=completed_boundary_stars,
        virtual_directions=virtual_count)
    return data,audit


class PiezoMessageGraph(torch.nn.Module):
    """Star energies of the plane piezoelectric lattice and their gradient (action) on a fixed lattice."""

    def __init__(self,mesh,material=None,thickness=.001,parameter_scales=None,
                 dtype=torch.float64,device='cpu'):
        super().__init__()
        if dtype not in (torch.float32,torch.float64):
            raise ValueError('A real floating Torch dtype is required')
        thickness = float(thickness)
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError('Positive finite thickness required')
        physical = _physical_parameters(material)
        _validate_physical(physical)
        if parameter_scales is not None and np.iscomplexobj(np.asarray(parameter_scales)):
            raise ValueError('Parameter scales must be real')
        scales = DEFAULT_SCALES.copy() if parameter_scales is None else np.asarray(parameter_scales,float)
        if scales.shape != (9,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError('Nine positive finite parameter scales required')
        data,audit = _geometry(mesh,thickness)
        self.n = len(data['coordinates']); self.ndof = 3*self.n
        self.nstar = len(data['star_scale']); self.nbond = len(data['bond_star'])
        self.thickness,self.spacing = thickness,float(mesh.spacing)
        for name,value in data.items():
            integer = np.issubdtype(value.dtype,np.integer)
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.long if integer else dtype,device=device))
        self.register_buffer('parameter_scales',torch.as_tensor(scales,dtype=dtype,device=device))
        self.register_buffer('initial_theta',torch.as_tensor(physical/scales,dtype=dtype,device=device))
        kind = data['bond_kind']
        for name,value in dict(axial=kind<4,diagonal=kind>=4,
                               horizontal=kind<2,vertical=(kind>=2)&(kind<4)).items():
            self.register_buffer(name,torch.as_tensor(value,dtype=dtype,device=device))
        self.metadata = dict(model='plane direct-star lattice with tip sectors',
            particles=self.n,stars=self.nstar,scalar_directions=self.nbond,
            scalar_bond_node_incidence=len(data['coefficient_value']),
            parameter_names=list(PARAMETER_NAMES),**audit)

    @property
    def source_graph(self):
        """Bipartite incidence buffers, not global material-matrix entries."""
        return dict(node=self.coefficient_node,bond=self.coefficient_bond,
                    coefficient=self.coefficient_value,bond_star=self.bond_star,
                    star_center=self.star_center,star_cell=self.star_cell)

    def _check_theta(self,theta):
        if (not isinstance(theta,torch.Tensor) or theta.shape != (9,)
                or theta.dtype != self.parameter_scales.dtype or theta.device != self.parameter_scales.device):
            raise ValueError('Theta must match nine material coordinates, dtype and device')

    def validate_parameters(self,theta):
        """Explicit once-per-material scalar check; not invoked within matvec."""
        self._check_theta(theta)
        _validate_physical((theta*self.parameter_scales).detach().cpu().numpy())

    def _state(self,state):
        if (not isinstance(state,torch.Tensor) or state.numel() != self.ndof
                or state.dtype != self.parameter_scales.dtype or state.device != self.parameter_scales.device):
            raise ValueError('State must match particle count, dtype and device')
        return state.reshape(self.n,3)

    def _material_coefficients(self,theta):
        self._check_theta(theta)
        c11,c22,c12,c66,e15,e31,e33,k11,k22 = theta*self.parameter_scales
        spring = ((c11-c12)/2*self.horizontal+(c22-c12)/2*self.vertical+c66/2*self.diagonal)
        return spring,(c12-c66)/12,((e33-e31)/2*self.axial+e15/2*self.diagonal), \
            (e31-e15)/4,(k11*self.horizontal+k22*self.vertical)/2

    def _differences(self,state):
        state = self._state(state)
        delta = state.new_zeros((self.nbond,3)).index_add(
            0,self.coefficient_bond,self.coefficient_value[:,None]*state[self.coefficient_node])
        ell = (delta[:,:2]*self.bond_direction).sum(dim=1)
        return ell,delta[:,2]

    def _sum_star(self,values):
        return values.new_zeros(self.nstar).index_add(0,self.bond_star,values)

    def energy(self,state,theta):
        """Direct S/G/B scalar energy; independently defined, not x dot action/2."""
        spring,tv,eta,alpha,beta = self._material_coefficients(theta)
        ell,delta = self._differences(state)
        ny = self.bond_direction[:,1]
        sa,sd,p = self._sum_star(self.axial*ell),self._sum_star(self.diagonal*ell),self._sum_star(self.axial*ny*delta)
        local = self._sum_star(.5*spring*ell.square()+eta*ny*ell*delta-.5*beta*delta.square())
        return (self.star_scale*(local+.5*tv*(sa.square()+sd.square())+alpha*sa*p)).sum()

    def action(self,state,theta):
        """Analytic energy gradient: bond differences from the particle values, accumulated back to the particles."""
        spring,tv,eta,alpha,beta = self._material_coefficients(theta)
        ell,delta = self._differences(state)
        ny,si = self.bond_direction[:,1],self.bond_star
        sa,sd,p = self._sum_star(self.axial*ell),self._sum_star(self.diagonal*ell),self._sum_star(self.axial*ny*delta)
        fl = spring*ell+tv*(self.axial*sa[si]+self.diagonal*sd[si])+eta*ny*delta+alpha*self.axial*p[si]
        fp = -beta*delta+eta*ny*ell+alpha*self.axial*ny*sa[si]
        response = torch.cat((fl[:,None]*self.bond_direction,fp[:,None]),dim=1)*self.star_scale[si,None]
        return state.new_zeros((self.n,3)).index_add(
            0,self.coefficient_node,self.coefficient_value[:,None]*response[self.coefficient_bond])

    def mechanical_diagonal(self,theta):
        """Exact uncontracted diag(K_uu)[n,2], including collective star terms."""
        spring,tv,_,_,_ = self._material_coefficients(theta)
        bi = self.coefficient_bond
        values = (self.coefficient_value[:,None]*self.bond_direction[bi]).square()
        values = values*(spring[bi]*self.star_scale[self.bond_star[bi]])[:,None]
        result = theta.new_zeros((self.n,2)).index_add(0,self.coefficient_node,values)
        collective = self.axial_node_coeff.square()+self.diagonal_node_coeff.square()
        collective = collective*(tv*self.star_scale[self.star_node_star])[:,None]
        return result.index_add(0,self.star_node,collective)

    def dielectric_diagonal(self,theta):
        """Exact positive dielectric magnitude -diag(K_phi_phi)[n], NOT signed K."""
        _,_,_,_,beta = self._material_coefficients(theta)
        bi = self.coefficient_bond
        value = beta[bi]*self.star_scale[self.bond_star[bi]]*self.coefficient_value.square()
        return theta.new_zeros(self.n).index_add(0,self.coefficient_node,value)
