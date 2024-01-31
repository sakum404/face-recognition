from getpass4 import getpass

def check_password():
    expected_password = "sakum"  # Замените на ваш реальный пароль
    entered_password = getpass("Введите пароль: ")

    if entered_password == expected_password:
        print("Пароль верный. Запуск программы.")
        return True
    else:
        print("Неверный пароль. Программа завершена.")
        return False

def main_program():
    # Ваш основной код программы
    print("Привет, это ваш основной код.")

if __name__ == "__main__":
    if check_password():
        main_program()