import copy
import dataclasses
import math
import random
from math import sqrt

import ezdxf
import numpy as np
from matplotlib import pyplot as plt


def get_solutions(solutions, start, end):
    if f"{start}-{end}" in solutions:
        return solutions[f"{start}-{end}"]
    else:
        return solutions[f"{end}-{start}"]


np.set_printoptions(suppress=True)

xfixed = ["0p"]
yfixed = ["0p", "9p"]
external = {
    "0p": [0, -2.25],
    "9p": [0, -2.25],
    "2p": [0, -4.5],
    "4p": [0, -4.5],
    "6p": [0, -4.5],
    "8p": [0, -4.5],
}

file = ezdxf.readfile("e(25).dxf")

msp = file.modelspace()


@dataclasses.dataclass()
class Point:
    x: float
    y: float

    def __hash__(self):
        return (round(self.x * 10), round(self.y * 10)).__hash__()

    def __eq__(self, other):
        return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01


@dataclasses.dataclass()
class Line:
    start: str
    end: str


def to_pnt(a):
    return Point(float(a[0]) * 3 + 7.44579, float(a[1]) * 3.5)


counter = 0
g_points = {}
connectivity = {}
lines = list(sorted(msp.query('LINE'), key=lambda k: k.dxf.start[0]))


def point_lookup(point) -> str:
    for p in g_points:
        if g_points[p] == point:
            return p


variables = []

for l in lines:
    startp, endp = to_pnt(l.dxf.start), to_pnt(l.dxf.end)
    plt.plot([startp.x, endp.x], [startp.y, endp.y], marker='o')

    start = point_lookup(startp)
    end = point_lookup(endp)

    if not start:
        g_points[f"{counter}p"] = startp
        connectivity[f"{counter}p"] = {}
        start = f"{counter}p"
        counter += 1
    if not end:
        g_points[f"{counter}p"] = endp
        connectivity[f"{counter}p"] = {}
        end = f"{counter}p"
        counter += 1

    sym = f"{start}-{end}"
    if sym == "4p-3p" or sym == "6p-7p":
        continue
    variables.append(sym)
    connectivity[start][end] = sym
    connectivity[end][start] = sym

for p in g_points:
    plt.text(g_points[p].x, g_points[p].y, f"{p}")


def print_node(solutions=None):
    fig = plt.figure(figsize=(10, 4), dpi=80)
    for p in g_points:
        plt.text(g_points[p].x, g_points[p].y, f"{p}", color="grey", fontsize='x-large')

    processed = set()
    for start in connectivity:
        for end in connectivity[start]:
            if (start, end) in processed:
                continue
            else:
                processed.add((start, end))
            startp = g_points[start]
            endp = g_points[end]
            if solutions:
                force = get_solutions(solutions, start, end)
                color = 'red' if force < 0 else 'blue'
                avg_coord_x = (startp.x + endp.x) / 2 - 0.3
                avg_coord_y = (startp.y + endp.y) / 2 - 0.1

                plt.plot([startp.x, endp.x], [startp.y, endp.y], color=color)
                plt.text(avg_coord_x, avg_coord_y, f"{force:.2f}kN")
            else:
                plt.plot([startp.x, endp.x], [startp.y, endp.y])

    plt.axis("scaled")
    fig.savefig("bridge1.png", dpi=200)
    fig.show()


print_node()


def append_if(list, elem):
    if elem not in list:
        list.append(elem)


np.set_printoptions(linewidth=np.inf)


def bmatrix(a, vars, eqns):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    a = a.round(2)
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')

    for v in range(0, len(vars)):
        vars[v] = r'\textbf{' + vars[v] + '}'
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{blockarray}' + '{' + 'c' * (a.shape[1] + 1) + '}']
    rv += [' ' + ' & '.join(vars) + r'\\']
    rv += [r'\begin{block}{(' + 'c' * a.shape[1] + ')c}']
    eqns1 = []
    for e in eqns:
        eqns1.append(r'\textbf{' + e + '}' + '_x')
        eqns1.append(r'\textbf{' + e + '}' + '_y')
    eqns1 += [r'\textbf{Total}_x']
    eqns1 += [r'\textbf{Total}_y']
    eqns1 += [r'\textbf{Total}_{Moment}']

    rv += ['  ' + ' & '.join(l.split()) + ' & ' + eqn + r' \\' for l, eqn in zip(lines, eqns1)]
    rv += [r'\end{blockarray}']
    return '\n'.join(rv)


def bmatrix1(a):
    a = a.round(2)
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')

    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r' \\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


def resolve(points):
    eqns = []

    extx = {"const": 0}
    exty = {"const": 0}
    ext_moment = {"const": 0}
    for start in connectivity:
        startp = points[start]
        sum_x = {}
        sum_y = {}
        for end in connectivity[start]:
            endp = points[end]
            dx = endp.x - startp.x
            dy = endp.y - startp.y
            length = sqrt(dx ** 2 + dy ** 2)

            if start < end:
                print(f"{start}-{end} length: ", length)

            ratiox = dx / length
            ratioy = dy / length

            sum_x[connectivity[start][end]] = ratiox
            sum_y[connectivity[start][end]] = ratioy

        if start in xfixed:
            symx = f"fixed-{start}-x"
            append_if(variables, symx)
            sum_x[symx] = 1
            extx[symx] = 1
            ext_moment[symx] = -startp.y
        if start in yfixed:
            symy = f"fixed-{start}-y"
            append_if(variables, symy)
            sum_y[symy] = 1
            exty[symy] = 1
            ext_moment[symy] = startp.x
        if start in external:
            diff_x, diff_y = external[start]
            sum_y["const"] = -diff_y
            sum_x["const"] = -diff_x
            extx["const"] += -diff_x
            exty["const"] += -diff_y
            ext_moment["const"] += -np.cross([startp.x, startp.y], [diff_x, diff_y])
        else:
            sum_y["const"] = 0
            sum_x["const"] = 0
        eqns.append(sum_x)
        eqns.append(sum_y)

    eqns.extend([extx, exty, ext_moment])

    for eqn in eqns:
        for var in variables:
            if var not in eqn:
                eqn[var] = 0
    variables_sorted = list(sorted(variables))
    matrix = []
    B = []
    for eqn in eqns:
        li = []
        for var in variables_sorted:
            if abs(eqn[var]) > 0.0001:
                li.append(eqn[var])
            else:
                li.append(0)
        B.append(eqn["const"])
        matrix.append(li)
    matrix = np.matrix(matrix)
    B = np.matrix(B).T
    print(bmatrix(matrix,  variables_sorted, list(connectivity.keys()),))
    print(bmatrix1(B.T))
    sln = np.linalg.lstsq(matrix, B, rcond=None)[0]
    a = np.square(np.matmul(matrix, sln) - B)
    print("Error: ", np.sum(a))
    sln = sln.T.A1
    print(bmatrix1(sln))
    solutions = {k: v for k, v in zip(variables_sorted, sln)}

    return solutions, sln, np.sum(a)


g_points = {'0p': Point(x=-0.017292870822722683, y=0.0),
            '1p': Point(x=4.729417074525266, y=2.3587279211271275),
            '2p': Point(x=2.982707129177279, y=0.0),
            '3p': Point(x=2.7170685066633924, y=-1.8055126620572644),
            '4p': Point(x=5.982707129177278, y=0.0),
            '5p': Point(x=10.055548481616082, y=2.5221979580774407),
            '6p': Point(x=8.98270712917728, y=0.0),
            '7p': Point(x=12.11374080689809, y=-1.5693453527929317),
            '8p': Point(x=11.982707129177278, y=0.0),
            '9p': Point(x=14.982707129177278, y=-1.2e-15)}


sln = resolve(g_points)[0]
print(sln)
# print_node(sln)
exit(0)
plt.figure()
cmap = plt.get_cmap("Reds")

point_keys = list(g_points.keys())
plt.show()
forbidden = set(xfixed + list(external.keys()))
allowed = set(point_keys).difference(forbidden)


def get_max_keys():
    return random.choices(list(allowed), k=2)


def get_cost(points, connectivity):
    length = 0
    for start in connectivity:
        for end in connectivity[start]:
            startp = points[start]
            endp = points[end]
            length += sqrt((startp.x - endp.x) ** 2 + (startp.y - endp.y) ** 2)
    return length / 2


best_cost = 99999999
stagnancy = 0
while True:
    points = copy.deepcopy(g_points)
    key1 = get_max_keys()

    factor = 0.05
    if stagnancy > 200:
        factor = 1.5

    for k in key1:
        points[k].x += (random.random() - 0.5) * factor
        points[k].y += (random.random() - 0.5) * factor

    if get_cost(points, connectivity) > best_cost:
        continue

    solutions, matrix, error = resolve(points)

    if (
            error < 1e-10
            and get_cost(points, connectivity) < best_cost
            and np.all((matrix[:-4] <= 12.01) & (matrix[:-4] >= -9.01))
    ):
        stagnancy = 0
        g_points = points
        best_cost = get_cost(points, connectivity)
        # best_cost = np.max(np.abs(matrix[:-4]))

        if random.random() < 0.01 or stagnancy > 400 or best_cost < 62.1:
            print(best_cost, solutions, matrix, g_points)
            print_node()
        # max_ = np.max(matrix)
        # processed = set()
        # for start in connectivity:
        #     for end in connectivity[start]:
        #         if connectivity[start][end] in processed:
        #             continue
        #
        #         processed.add(connectivity[start][end])
        #
        #         startp, endp = g_points[start], g_points[end]
        #         force = get_solutions(solutions, start, end) / max_
        # ax.plot([startp.x, endp.x], [startp.y, endp.y], marker='o', color=cmap(force))
        # for p in g_points:
        #     ax.text(g_points[p].x, g_points[p].y, f"{p}")
        # fig.show()
    else:
        stagnancy += 1
    # plt.show()
# print(solve(eqns[3:], warn=True))
