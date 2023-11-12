import pygame
from sklearn import svm


RADIUS = 5


def add_new_point(screen, color, position, points, classes, class_number=0):
    """
    Метод добавляет новую точку, отображает её и записывает в массив точек
    :param screen:
    :param color:
    :param position:
    :param points:
    :param classes:
    :param class_number:
    :return:
    """
    pygame.draw.circle(screen, color=color, center=position, radius=RADIUS)
    pygame.display.update()
    points.append(list(position))
    classes.append(class_number)


def add_new_point_with_class_predict(model, position, screen, colors):
    """
    Метод предсказывает класс для точки и отображает её соответствующим цветом
    :param model:
    :param position:
    :param screen:
    :param colors:
    :return:
    """
    class_predicted = model.predict([position])
    pygame.draw.circle(screen, color=colors[class_predicted[0]], center=position, radius=RADIUS)
    pygame.display.update()


def add_grade_separation_line(points, classes, model, screen):
    """
    Метод отображает прямую, разделяющую классы
    :param points:
    :param classes:
    :param model:
    :param screen:
    :return:
    """
    model.fit(points, classes)
    coef = model.coef_[0]
    start_pos = [0, model.intercept_[0] / -coef[1]]
    end_pos = [800, coef[0] / -coef[1] * 800 + model.intercept_[0] / -coef[1]]
    pygame.draw.line(screen, color='black', start_pos=start_pos, end_pos=end_pos)
    pygame.display.update()

def start_svm_algorithm():
    """
    Метод запускаем алгоритм SVM. Алгоритм переходит в режим обучения.
    Открывается окно для отрисовки точек. Левой кнопкой выбираются точки одного класса, правой - другого класса.
    После нажатия ENTER алгоритм отобразит прямую, которая разделить множества. Алгоритм переходит в режим тестирования.
    Последующие добавленные точки будут присваиваться соответствующим кластерам.
    :return:
    """
    model = svm.SVC(kernel='linear')
    # Все наши точки, которые мы добавим на экране
    points = []
    # Класс, которому принадлежит точка индекса (0 или 1)
    classes = []
    play = True
    learning_mode = True
    colors = ['red', 'blue', 'black', 'yellow']

    pygame.init()
    screen = pygame.display.set_mode([800, 600])
    screen.fill(color='white')
    pygame.display.update()

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if learning_mode:
                    add_new_point(screen, colors[0], event.pos, points, classes)
                else:
                    add_new_point_with_class_predict(model, event.pos, screen, colors)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if learning_mode:
                    add_new_point(screen, colors[1], event.pos, points, classes, class_number=1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                learning_mode = False
                add_grade_separation_line(points, classes, model, screen)



start_svm_algorithm()