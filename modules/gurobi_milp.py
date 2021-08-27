from gurobipy import *
from math import *
import numpy as np
from collections import namedtuple
import time

M = 1e3
# a large M causes numerical issues and make the model infeasible to Gurobi
T_MIN_SEP = 1e-2
# see comments in GreaterThanZero
IntFeasTol  = 1e-1 * T_MIN_SEP / M

def setM(v):
    global M, IntFeasTol
    M = v
    IntFeasTol  = 1e-1 * T_MIN_SEP / M

EPS = 1e-4

def _sub(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]

def _add(x1, x2):
    return [x1[i] + x2[i] for i in range(len(x1))]

def L1Norm(model, x):
    xvar = model.addVars(len(x), lb=-GRB.INFINITY)
    abs_x = model.addVars(len(x))
    model.update()
    xvar = [xvar[i] for i in range(len(xvar))]
    abs_x = [abs_x[i] for i in range(len(abs_x))]
    for i in range(len(x)):
        model.addConstr(xvar[i] == x[i])
        model.addConstr(abs_x[i] == abs_(xvar[i]))
    return sum(abs_x)

class Conjunction(object):
    # conjunction node
    def __init__(self, deps = []):
        super(Conjunction, self).__init__()
        self.deps = deps
        self.constraints = []

class Disjunction(object):
    # disjunction node
    def __init__(self, deps = []):
        super(Disjunction, self).__init__()
        self.deps = deps
        self.constraints = []

def noIntersection(a, b, c, d):
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b < c or d < a
    return Disjunction([c-b-EPS, a-d-EPS])

def hasIntersection(a, b, c, d):
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b >= c and d >= a
    return Conjunction([b-c, d-a])

def always(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    conjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        conjunctions.append(Disjunction([noIntersection(t_j, t_j_1, t_i + a, t_i_1 + b), zphis[j]]))
    return Conjunction(conjunctions)

def eventually(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    z_intervalWidth = b-a-(t_i_1-t_i)-EPS
    disjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        disjunctions.append(Conjunction([hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphis[j]]))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])

def bounded_eventually(i, a, b, zphis, PWL, tmax):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    z_intervalWidth = b-a-(t_i_1-t_i)-EPS
    disjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        disjunctions.append(Conjunction([hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphis[j]]))
    return Disjunction([Conjunction([z_intervalWidth, Disjunction(disjunctions)]), t_i+b-tmax-EPS])

def until(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    z_intervalWidth = b-a-(t_i_1-t_i)-EPS
    disjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        conjunctions = [hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j+1):
            t_l = PWL[l][1]
            t_l_1 = PWL[l+1][1]
            conjunctions.append(Disjunction([noIntersection(t_l, t_l_1, t_i, t_i_1 + b), zphi1s[l]]))
        disjunctions.append(Conjunction(conjunctions))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])

def release(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    conjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        disjunctions = [noIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j):
            t_l = PWL[l][1]
            t_l_1 = PWL[l+1][1]
            disjunctions.append(Conjunction([hasIntersection(t_l, t_l_1, t_i_1, t_i_1 + b), zphi1s[l]]))
        conjunctions.append(Disjunction(disjunctions))
    return Conjunction(conjunctions)

def mu(i, PWL, bloat_factor, A, b):
    # this segment is fully contained in Ax<=b (shrinked)
    b = b.reshape(-1)
    num_edges = len(b)
    conjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        for j in [i, i+1]:
            x = PWL[j][0]
            conjunctions.append(b[e] - np.linalg.norm(a) * bloat_factor - sum([a[k]*x[k] for k in range(len(x))]) - EPS)
    return Conjunction(conjunctions)

def negmu(i, PWL, bloat_factor, A, b):
    # this segment is outside Ax<=b (bloated)
    b = b.reshape(-1)
    num_edges = len(b)
    disjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        conjunctions = []
        for j in [i, i+1]:
            x = PWL[j][0]
            conjunctions.append(sum([a[k]*x[k] for k in range(len(x))]) - (b[e] + np.linalg.norm(a) * bloat_factor) - EPS)
        disjunctions.append(Conjunction(conjunctions))
    return Disjunction(disjunctions)

def add_space_constraints(model, xlist, limits, bloat=0.):
    for x in xlist:
        for i in range(len(x)):
            model.addConstr(x[i] >= (limits[i][0] + bloat))
            model.addConstr(x[i] <= (limits[i][1] - bloat))

def add_time_constraints(model, PWL, tmax=None):
    if tmax is not None:
        model.addConstr(PWL[-1][1] <= tmax - T_MIN_SEP)

    for i in range(len(PWL)-1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i+1]
        model.addConstr(t2 - t1 >= T_MIN_SEP)

def add_velocity_constraints(model, PWL, vmax=3):
    for i in range(len(PWL)-1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i+1]
        # squared_dist = sum([(x1[j]-x2[j])*(x1[j]-x2[j]) for j in range(len(x1))])
        # model.addConstr(squared_dist <= (vmax**2) * (t2 - t1) * (t2 - t1))
        L1_dist = L1Norm(model, _sub(x1,x2))
        model.addConstr(L1_dist <= vmax * (t2 - t1))

def disjoint_segments(model, seg1, seg2, bloat):
    assert(len(seg1) == 2)
    assert(len(seg2) == 2)
    # assuming that bloat is the error bound in two norm for one agent
    return 0.5 * L1Norm(model, _sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1]))) - 0.5 * (L1Norm(model, _sub(seg1[0], seg1[1])) + L1Norm(model, _sub(seg2[0], seg2[1]))) - 2*bloat*np.sqrt(len(seg1[0])) - EPS

def add_mutual_clearance_constraints(model, PWLs, bloat):
    for i in range(len(PWLs)):
        for j in range(i+1, len(PWLs)):
            PWL1 = PWLs[i]
            PWL2 = PWLs[j]
            for k in range(len(PWL1)-1):
                for l in range(len(PWL2)-1):
                    x11, t11 = PWL1[k]
                    x12, t12 = PWL1[k+1]
                    x21, t21 = PWL2[l]
                    x22, t22 = PWL2[l+1]
                    z_noIntersection = noIntersection(t11, t12, t21, t22)
                    z_disjoint_segments = disjoint_segments(model, [x11, x12], [x21, x22], bloat)
                    z = Disjunction([z_noIntersection, z_disjoint_segments])
                    add_CDTree_Constraints(model, z)

class Node(object):
    """docstring for Node"""
    def __init__(self, op, deps = [], zs = [], info = []):
        super(Node, self).__init__()
        self.op = op
        self.deps = deps
        self.zs = zs
        self.info = info

def clearSpecTree(spec):
    for dep in spec.deps:
        clearSpecTree(dep)
    spec.zs = []

def handleSpecTree(spec, PWL, bloat_factor):
    for dep in spec.deps:
        handleSpecTree(dep, PWL, bloat_factor)
    if len(spec.zs) == len(PWL)-1:
        return
    elif len(spec.zs) > 0:
        raise ValueError('incomplete zs')
    if spec.op == 'mu':
        spec.zs = [mu(i, PWL, bloat_factor, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
    elif spec.op == 'negmu':
        spec.zs = [negmu(i, PWL, bloat_factor, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
    elif spec.op == 'and':
        spec.zs = [Conjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWL)-1)]
    elif spec.op == 'or':
        spec.zs = [Disjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWL)-1)]
    elif spec.op == 'U':
        spec.zs = [until(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, spec.deps[1].zs, PWL) for i in range(len(PWL)-1)]
    elif spec.op == 'F':
        spec.zs = [eventually(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL) for i in range(len(PWL)-1)]
    elif spec.op == 'BF':
        spec.zs = [bounded_eventually(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL, spec.info['tmax']) for i in range(len(PWL)-1)]
    elif spec.op == 'A':
        spec.zs = [always(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL) for i in range(len(PWL)-1)]
    else:
        raise ValueError('wrong op code')

def gen_CDTree_constraints(model, root):
    if not hasattr(root, 'deps'):
        return [root,]
    else:
        if len(root.constraints)>0:
            # TODO: more check here
            return root.constraints
        dep_constraints = []
        for dep in root.deps:
            dep_constraints.append(gen_CDTree_constraints(model, dep))
        zs = []
        for dep_con in dep_constraints:
            if isinstance(root, Disjunction):
                z = model.addVar(vtype=GRB.BINARY)
                zs.append(z)
                dep_con = [con + M * (1 - z) for con in dep_con]
            root.constraints += dep_con
        if len(zs)>0:
            root.constraints.append(sum(zs)-1)
        model.update()
        return root.constraints

def add_CDTree_Constraints(model, root):
    constrs = gen_CDTree_constraints(model, root)
    for con in constrs:
        model.addConstr(con >= 0)

def plan_single_agent(x0, spec, bloat, limits=([-100., 100.], [-100., 100.]), num_segs=None, feasible_only=False, tmax=None, hard_tmax=None, hard_goal=None, vmax=3., MIPGap=1e-4):
    x0 = np.array(x0).reshape(-1).tolist()

    if num_segs is None:
        min_segs = 1
        max_segs = len(bloat)
    else:
        min_segs = num_segs
        max_segs = num_segs

    dims = len(x0)

    objs = []
    res = []
    for num_segs in range(min_segs, max_segs+1):
        clearSpecTree(spec)

        print('----------------------------')
        print('num_segs', num_segs)
        PWL = []
        m = Model("xref")
        m.setParam(GRB.Param.OutputFlag, 0)
        m.setParam(GRB.Param.IntFeasTol, IntFeasTol)
        m.setParam(GRB.Param.MIPGap, MIPGap)
        # m.setParam(GRB.Param.NonConvex, 2)
        # m.getEnv().set(GRB_IntParam_OutputFlag, 0)
        for i in range(num_segs+1):
            PWL.append([m.addVars(dims, lb=-GRB.INFINITY), m.addVar()])
        m.update()

        obj = PWL[-1][1] if not feasible_only else 0
        m.setObjective(obj, GRB.MINIMIZE)

        # the initial constriant
        m.addConstrs(PWL[0][0][i] == x0[i] for i in range(dims))
        m.addConstr(PWL[0][1] == 0)
        if hard_tmax is not None:
            m.addConstr(PWL[-1][1] == hard_tmax)
        if hard_goal is not None:
            for i in range(len(PWL[-1][0])):
                m.addConstr(PWL[-1][0][i] == hard_goal[i])

        add_space_constraints(m, [P[0] for P in PWL], limits)

        add_velocity_constraints(m, PWL, vmax=vmax)
        add_time_constraints(m, PWL, tmax)

        handleSpecTree(spec, PWL, bloat[0])
        add_CDTree_Constraints(m, spec.zs[0])

        # m.write("test.lp")

        print('NumBinVars: %d'%m.getAttr('NumBinVars'))
        try:
            start = time.time()
            m.optimize()
            end = time.time()
            print('sovling it takes %.3f s'%(end - start))
            PWL_output = []
            for P in PWL:
                PWL_output.append([[P[0][i].X for i in range(len(P[0]))], P[1].X])
            m.dispose()
            return PWL_output
        except Exception as e:
            m.dispose()
    return None


def plan_gurobi(obstacles, init, goal, limits, tmax=10., vmax=3., bloat_factor=1., num_segs=None, MIPGap=1e-4):
    avoid_obs = Node('and', deps=[Node('negmu', info={'A':A, 'b':b}) for A, b in obstacles])
    always_avoid_obs = Node('A', deps=[avoid_obs,], info={'int':[0,tmax]})
    spec = always_avoid_obs
    PWL = plan_single_agent(init, spec, [bloat_factor for _ in range(20)], num_segs=num_segs, limits=limits, hard_goal=goal, tmax=tmax, vmax=vmax, MIPGap=MIPGap)
    if PWL is None:
        return None
    else:
        return [P[0] for P in PWL]


def plan(x_init, waypoints, x_bound, obstacles, bloat_factor=2., num_segs=None, MIPGap=1e-4):
    curr = x_init
    path = [x_init,]
    for w in waypoints:
        #pert = 0.
        #PWL = None
        #while PWL is None:
        #    PWL = plan_gurobi(obstacles, curr, w, limits=x_bound, tmax=1e4, vmax=3., bloat_factor=bloat_factor+pert, num_segs=num_segs, MIPGap=MIPGap)
        #    pert = (np.random.rand()-0.5)*2.*1e-2
        PWL = plan_gurobi(obstacles, curr, w, limits=x_bound, tmax=1e4, vmax=3., bloat_factor=bloat_factor, num_segs=num_segs, MIPGap=MIPGap)
        if PWL is None:
            return None
        path += PWL[1:]
        curr = w
    path = np.array(path)
    return path # N x 3


def interpolate(wp_in, dt=0.1, max_speed=0.5):
    wp_out = []
    t = []
    currtent_t = 0.
    for i in range(wp_in.shape[0]-1):
        p1 = wp_in[i, :]
        p2 = wp_in[i+1, :]
        dist = np.sqrt(((p2 - p1)**2).sum())
        unit = (p2 - p1) / dist
        T = dist / max_speed
        local_t = np.arange(0., T, dt)
        wp_out += [p1+lt*max_speed*unit for lt in local_t]
        t += (local_t+currtent_t).tolist()
        currtent_t = t[-1] + dt
    t.append(currtent_t)
    wp_out.append(wp_in[-1,:])
    return np.array(wp_out), np.array(t)