QT += charts

INCLUDEPATH += . ./lib/eigenlib


HEADERS += \
    callout.h \
    view.h \
    kalman.h

SOURCES += \
    callout.cpp \
    main.cpp\
    view.cpp \
    kalman.cpp

target.path = $$[QT_INSTALL_EXAMPLES]/charts/callout
INSTALLS += target
