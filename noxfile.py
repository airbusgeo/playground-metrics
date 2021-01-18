import nox


@nox.session
def lint(session):
    session.install('.[lint]')
    session.run('flake8', '-v', 'playground_metrics')
    session.run('flake8', '-v', '--ignore=D', 'tests')


@nox.session(python=['3.6', '3.7', '3.8', '3.9'])
def tests(session):
    session.install('.[tests]')
    session.run('pytest', '-vv', '--cov-report', 'term-missing', '--cov=map_metric_api', 'tests/')
