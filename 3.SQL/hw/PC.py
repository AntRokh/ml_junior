import random


with open('create_and_fill_PC_.sql', 'w') as file:
    file.write(
        'CREATE TABLE PC_ (code INTEGER, model INTEGER, speed INTEGER, ram INTEGER, '
        'hd NUMERIC(10,1), cd VARCHAR(50), price NUMERIC(10,4));\n'
    )
    file.write('INSERT INTO PC_\n')
    file.write('VALUES\n')

    values = []
    for code, i in enumerate(range(1121, 1232), start=1):
        values.append(
            f'({code}, {i}, {random.randrange(500, 901, 100)},'
            f' {random.randrange(32, 129, 32)}, {random.randrange(5, 21, 5)},'
            f' \'{random.randrange(12, 53, 4)}x\', {random.randrange(350, 1001, 50)})'
        )

    # Чтобы в конце не стояла запятая и код корректно сработал
    file.write(',\n'.join(values) + ';\n')

