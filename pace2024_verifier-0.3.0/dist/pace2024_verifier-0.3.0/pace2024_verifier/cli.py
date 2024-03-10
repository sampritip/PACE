import click
import pathlib
import subprocess
import tempfile
import importlib.resources as irs
from .pace import read_graph, read_solution

@click.command()
@click.option('--interleave', 'method', flag_value='interleave', help='Count crossings checking pairs of edges one by one.')
@click.option('--stacklike', 'method', flag_value='stacklike', help='Count crossings using a stack-like method, similar to counting crossings in book drawings.')
@click.option('--segtree', 'method', flag_value='segtree', default=True, help='Count crossings using a segment tree. [default]')
@click.option('-c', '--only-crossings', is_flag=True, default=False, help='Print only the found number of crossings.')
@click.argument('graph', type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument('solution', type=click.Path(exists=True, path_type=pathlib.Path))
def verify(method, only_crossings, graph, solution):
    '''Print the number of crossings the given solution has.'''
    solution = read_solution(solution)
    graph = read_graph(graph, solution)

    if method == "interleave":
        crossings = graph.countcrossings_trivial()
    elif method == "stacklike":
        crossings = graph.countcrossings_stacklike()
    else:
        crossings = graph.countcrossings_segtree()

    if only_crossings:
        click.echo(crossings)
    else:
        click.echo(f"Using {solution} as ordering of the vertices we found {crossings} crossings using the {method} method.")

def solutionfromstdout(stdout):
    return list(map(int, stdout.strip().split("\n")))

def print_result(control: int, test: int):
    if control == test:
        click.echo(f"The solver and the solution are the same!")
    else:
        click.echo(f"The solver and the solution are NOT the same!")

    click.echo(f"Solver: {test} Solution: {control}")
    click.echo()

def print_test(instance: pathlib.Path):
    click.echo(f"Testing now {instance[0].name}...")

@click.command()
@click.option('--tiny/--no-tiny', default=True, help="If set run the tests on the tiny test set.")
@click.option('--test', type=click.Path(exists=True, path_type=pathlib.Path, file_okay=False), multiple=True, default=None, help="Add additional test sets. Can be supplied multiple times of directories, each of them should contain a folder instances with .gr files and a folder solutions containing .sol files with the same file names else.")
@click.option('--instanceas', type=click.Choice(['file', 'stdin']), default='file', help="How the instance is passed to the solver, either as a path to a file or on stdin.")
@click.option('--solutionas', type=click.Choice(['file', 'stdout']), default='file', help="How the soltution is returned by the solver. Either written to a file, we provide a temporary file path, or to stdout.")
@click.option('-c', '--only-compare', is_flag=True, default=False, help='Print only the number of crossings by the solver and in the solution.')
@click.argument('solver', type=str)
def test(tiny, test, instanceas, solutionas, only_compare, solver):
    '''Test the given solver on the provided test instances.'''

    instances = [] # Collect the set of instances as a list of tuples

    if tiny:
        tinytests = irs.files("pace2024_verifier.tiny_test_set")
        instances += list(zip(tinytests.joinpath("instances").iterdir(),tinytests.joinpath("solutions").iterdir()))

    if test:
        for testdir in test:
            instances += list(zip(testdir.joinpath("instances").iterdir(),testdir.joinpath("solutions").iterdir()))

    if solutionas == 'file':
        with tempfile.TemporaryDirectory() as tmp:        
            for instance in instances:
                print_test(instance)

                tmpsolution = pathlib.Path(tmp).joinpath(f"{instance[1].stem}.sol")
                testsolution = None

                if instanceas == 'file':
                    ret = subprocess.run([solver, instance[0], tmpsolution], text=True, capture_output=True)
                    testsolution = read_solution(tmpsolution)
                else:
                    graph = read_graph(instance[0])
                    ret = subprocess.run([solver, tmpsolution], input=graph.to_gr(), text=True, capture_output=True)
                    testsolution = read_solution(tmpsolution)

                controlsolution = read_solution(instance[1])

                graph = read_graph(instance[0], controlsolution)
                controlcrossings = graph.countcrossings_segtree()
                graph.set_order(testsolution)
                testcrossings = graph.countcrossings_segtree()

                if not only_compare:
                    print_result(controlcrossings, testcrossings)
                else:
                    click.echo(f"{testcrossings},{controlcrossings}")
    else:
        for instance in instances:
            print_test(instance)

            testsolution = None

            if instanceas == 'file':
                ret = subprocess.run([solver, instance[0]], text=True, capture_output=True)
                testsolution = solutionfromstdout(ret.stdout)
            else:
                graph = read_graph(instance[0])
                ret = subprocess.run([solver], input=graph.to_gr(), text=True, capture_output=True)
                testsolution = solutionfromstdout(ret.stdout)

            controlsolution = read_solution(instance[1])

            graph = read_graph(instance[0], controlsolution)
            controlcrossings = graph.countcrossings_segtree()
            graph.set_order(testsolution)
            testcrossings = graph.countcrossings_segtree()

            if not only_compare:
                print_result(controlcrossings, testcrossings)
            else:
                click.echo(f"{testcrossings},{controlcrossings}")
            
