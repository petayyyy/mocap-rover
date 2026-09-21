# Hardware backlog

Открыто после S01; аппаратная фаза не запускалась.

- Проверить допустимость потолочной сетки, центральной опоры для
  `hybrid_research` и верхних perimeter-точек.
- Подтвердить конкретную оптику, image circle, distortion, edge resolution,
  корпус и монтаж для IMX296.
- Определить транспорт, узлы захвата, trigger и аппаратную синхронизацию.
- Получить реальные CPU/GPU/VRAM и bandwidth measurements на целевом стенде.
- Проверить правила площадки, освещение/ИК и реальные occlusion/заслонения.
- После SIM_ACCEPTED отдельно проверить реальный AprilTag/YOLO runtime и
  domain gap; текущая OpenCV AprilTag probe завершилась native exit 139 и не
  является аппаратной проверкой.
- Зафиксировать реальный dataset/лицензии/веса YOLO, выдержку, exposure/WB и
  измеренные CPU/GPU/VRAM/bandwidth на целевом стенде.
- Повторить timing/trigger/skew/transport measurements на физических узлах;
  sim/replay clock models не подтверждают PTP/NTP/trigger.

Эти вопросы не блокируют S01 и SIM-фазу; H01–H04 начинаются только после S15.
