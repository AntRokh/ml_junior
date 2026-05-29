import random
import string

with open('create_and_fill_Product_.sql', 'w') as file:
    file.write(
        'CREATE TABLE Product_ (maker VARCHAR(50), model INTEGER, type VARCHAR(50));\n'
    )
    file.write('INSERT INTO Product_\n')
    file.write('VALUES\n')
    values = []

    for i in range(1121, 1232):
        values.append(
            f"('{string.ascii_uppercase[random.randint(0, 5)]}', {i}, 'PC')"
        )
    for i in range(1232, 1360):
        values.append(
            f"('{string.ascii_uppercase[random.randint(0, 5)]}', {i}, 'Laptop')"
        )
    for i in range(1360, 1484):  # random_letter = string.ascii_uppercase[random.randint(0, 25)]
        values.append(
            f"('{string.ascii_uppercase[random.randint(0, 5)]}', {i}, 'Printer')"
        )
    # Чтобы в конце не стояла запятая и код корректно сработал
    file.write(',\n'.join(values) + ';\n')