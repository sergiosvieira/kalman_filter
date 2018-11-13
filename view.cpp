/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the Qt Charts module of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:GPL$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3 or (at your option) any later version
** approved by the KDE Free Qt Foundation. The licenses are as published by
** the Free Software Foundation and appearing in the file LICENSE.GPL3
** included in the packaging of this file. Please review the following
** information to ensure the GNU General Public License requirements will
** be met: https://www.gnu.org/licenses/gpl-3.0.html.
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "view.h"
#include <QtGui/QResizeEvent>
#include <QtWidgets/QGraphicsScene>
#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QtCharts/QSplineSeries>
#include <QtWidgets/QGraphicsTextItem>
#include "callout.h"
#include <QtGui/QMouseEvent>
#include <Eigen/Dense>
#include "kalman.h"

View::View(QWidget *parent)
    : QGraphicsView(new QGraphicsScene, parent),
      m_coordX(0),
      m_coordY(0),
      m_chart(0),
      m_tooltip(0)
{
    int n = 9; // Number of states
    int m = 3; // Number of measurements
    unsigned int dt = 5.; // delta
    unsigned int endTime = 60;
    Eigen::MatrixXd A(n, n); // System dynamics matrix (Transformation Matrix)
    Eigen::MatrixXd C(m, n); // Output matrix
    Eigen::MatrixXd Q(n, n); // Process noise covariance
    Eigen::MatrixXd R(m, m); // Measurement noise covariance
    Eigen::MatrixXd P(n, n); // Estimate error covariance
    // Discrete LTI projectile motion, measuring position only
    double tt = 0.5 * dt * dt;
    A << 1, 0, 0, dt,  0,  0, tt,  0,  0,
         0, 1, 0,  0, dt,  0,  0, tt,  0,
         0, 0, 1,  0,  0, dt,  0,  0, dt,
         0, 0, 0,  1,  0,  0, dt,  0,  0,
         0, 0, 0,  0,  1,  0,  0, dt,  0,
         0, 0, 0,  0,  0,  1,  0,  0, dt,
         0, 0, 0,  0,  0,  0,  1,  0,  0,
         0, 0, 0,  0,  0,  0,  0,  1,  0,
         0, 0, 0,  0,  0,  0,  0,  0,  1;
    C << 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0;
    // Reasonable covariance matrices
    Q << tt, .05, .05, .0, .0, .0, .0, .0, .0,
         .05, tt, .05, .0, .0, .0, .0, .0, .0,
         .05, .5, tt, .0, .0, .0, .0, .0, .0,
         .0, .0, .0, .0, .0, .0, .0, .0, .0,
         .0, .0, .0, .0, .0, .0, .0, .0, .0,
         .0, .0, .0, .0, .0, .0, .0, .0, .0,
         .0, .0, .0, .0, .0, .0, .0, .0, .0,
         .0, .0, .0, .0, .0, .0, .0, .0, .0,
         .0, .0, .0, .0, .0, .0, .0, .0, .0;
    R << 5, 0, 0,
         0, 5, 0,
         0, 0, 5;
    P << 1.3, 2, 3, 1, 1, 1, 1, 1, 1,
         4, 1.5, 5, 1, 1, 1, 1, 1, 1,
         6, 7, 9, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1.6, 1, 1, 1, 1, 1,
         1, 1, 1, 1, .9, 1, 1, 1, 1,
         1, 1, 1, 1, 1, .2, 1, 1, 1,
         1, 1, 1, 1, 1, 1, .1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1.54, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1.2;
    KalmanFilter k(dt, A, C, Q, R, P);
    Eigen::VectorXd s0(n), y(m);
    s0 <<  7.0,  8.0,  9.0, 1.0,  2.0,  3.0, 0.33, 0.66, 0.99;
    //      px,   py,   pz,  vx,   vy,   vz,   ax,   ay,   az
    k.init(0, s0);

    setDragMode(QGraphicsView::NoDrag);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // chart
    m_chart = new QChart;
    m_chart->setMinimumSize(640, 480);
    m_chart->setTitle("Distance to Origin (m) x Instant (s)");
    //m_chart->legend()->hide();
    m_chart->legend()->setVisible(true);
    m_chart->legend()->setAlignment(Qt::AlignBottom);


    QLineSeries *series = new QLineSeries;
    series->setName("Observed");
    QSplineSeries *series2 = new QSplineSeries;
    series2->setName("Predicted");
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);
    for (unsigned int t = 0; t <= endTime; t += dt)
    {
        Eigen::VectorXd u = Eigen::VectorXd::Random(n);
        Eigen::VectorXd y_ = A * s0 + B * u;
        Eigen::Vector3d v;
        v << y_[0], y_[1], y_[2];
        k.update(v);
        series->append(t, v.norm());// distance to origin
        series2->append(t, k.state().norm()); // distance to origin
        s0 = y_;
    }
    m_chart->addSeries(series);
    m_chart->addSeries(series2);

    m_chart->createDefaultAxes();
    m_chart->setAcceptHoverEvents(true);

    setRenderHint(QPainter::Antialiasing);
    scene()->addItem(m_chart);

    m_coordX = new QGraphicsSimpleTextItem(m_chart);
    m_coordX->setPos(m_chart->size().width()/2 - 50, m_chart->size().height());
    m_coordX->setText("X: ");
    m_coordY = new QGraphicsSimpleTextItem(m_chart);
    m_coordY->setPos(m_chart->size().width()/2 + 50, m_chart->size().height());
    m_coordY->setText("Y: ");

    connect(series, &QLineSeries::clicked, this, &View::keepCallout);
    connect(series, &QLineSeries::hovered, this, &View::tooltip);

    connect(series2, &QSplineSeries::clicked, this, &View::keepCallout);
    connect(series2, &QSplineSeries::hovered, this, &View::tooltip);

    this->setMouseTracking(true);
}

void View::resizeEvent(QResizeEvent *event)
{
    if (scene()) {
        scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
         m_chart->resize(event->size());
         m_coordX->setPos(m_chart->size().width()/2 - 50, m_chart->size().height() - 20);
         m_coordY->setPos(m_chart->size().width()/2 + 50, m_chart->size().height() - 20);
         const auto callouts = m_callouts;
         for (Callout *callout : callouts)
             callout->updateGeometry();
    }
    QGraphicsView::resizeEvent(event);
}

void View::mouseMoveEvent(QMouseEvent *event)
{
    m_coordX->setText(QString("X: %1").arg(m_chart->mapToValue(event->pos()).x()));
    m_coordY->setText(QString("Y: %1").arg(m_chart->mapToValue(event->pos()).y()));
    QGraphicsView::mouseMoveEvent(event);
}

void View::keepCallout()
{
    m_callouts.append(m_tooltip);
    m_tooltip = new Callout(m_chart);
}

void View::tooltip(QPointF point, bool state)
{
    if (m_tooltip == 0)
        m_tooltip = new Callout(m_chart);

    if (state) {
        m_tooltip->setText(QString("X: %1 \nY: %2 ").arg(point.x()).arg(point.y()));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}
