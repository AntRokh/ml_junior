import random


with open('create_and_fill_Printer_.sql', 'w') as file:
    file.write(
        'CREATE TABLE Printer_ (code INTEGER, model INTEGER, color char,' #Printer (code, model, color, type, price).
        'type VARCHAR(25), price NUMERIC(10,4));\n'
    )
    file.write('INSERT INTO Printer_\n')
    file.write('VALUES\n')

    values = []
    for code, i in enumerate(range(1361, 1484), start=1):
        values.append(
            f"({code}, {i}, '{random.choice(['y', 'n'])}',"
            f" '{random.choice(['Laser', 'Matrix', 'Jet'])}', {random.randrange(350, 1001, 50)})"
        )

    # Чтобы в конце не стояла запятая и код корректно сработал
    file.write(',\n'.join(values) + ';\n')

