package com.wicri.mine.ids.algorithm.misc.decision;

import com.wicri.mine.ids.algorithm.misc.decision.util.GetNearestPoint;
import com.wicri.mine.ids.algorithm.misc.decision.util.Queue;
import com.wicri.mine.ids.algorithm.misc.decision.pojo.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * @author ：fzq.
 * @date ：Created in 9:11 2021/6/30
 * @description：决策防碰撞算法
 */
public class DecisionAlgorithm {
    //排队间隔点数量
    public static Integer queueIntervalPointNum = 5;

    //车辆当前所在区域的后一个区域信息
    private volatile TheArea lastTheAreaInfo = null;

    //局部路径
    List<Location> localPath = new ArrayList<>();

    //所有工作区域间的距离集合
    private List<AreaDistance> areaDistanceList;

    public Result decisionAlgorithm(Car car, List<TheArea> theAreas) {
        localPath = new ArrayList<>();
        Result result = new Result();

        Location carLocation = car.getCarLocation();
        for (TheArea theArea : theAreas) {
            if (RegionalOrientationAlgorithm.rayAlgorithm(carLocation, theArea.getBoundaryLocation())) {
                car.setLocalTheArea(theArea);
                break;
            }
        }


        List<Integer> directedAreaIdList = new ArrayList<>();
        for (Long aLong : car.getDirectedAreaIdList()) {
            directedAreaIdList.add(Integer.parseInt((aLong.toString())));
        }
        int preAreaId = -1;
        int lastAreaId = -1;
        //根据有向图和车辆当前所在区域获取前一个区域id和后一个区域id
        for (int i = 0; i < directedAreaIdList.size(); i++) {
            if (directedAreaIdList.get(i).toString().equals(car.getLocalTheArea().getAreaId().toString())) {
                if (i > 0) {
                    preAreaId = directedAreaIdList.get(i - 1);
                }
                if (i < directedAreaIdList.size() - 1) {
                    lastAreaId = directedAreaIdList.get(i + 1);
                }
                break;
            }
        }

        TheArea area = new TheArea();
        //获取当前所在区域的后一个区域信息
        for (TheArea theArea : theAreas) {
            if (theArea.getAreaId() == lastAreaId) {
                area = theArea;
                break;
            }
        }
        lastTheAreaInfo = area;

        //当前车辆出工作区域或者路口后修改对应区域状态;
        //遍历区域集合,根据前一个区域id获取前一个区域信息;
        for (TheArea theArea : theAreas) {
            if (preAreaId != -1 && theArea.getAreaId() == preAreaId) {
                //判断该区域 前一个区域不为空, 前一个区域状态已占用, 前一个区域占用id == 当前车辆id
                if (theArea.getAreaState() == 1 && theArea.getOccupyCar().getCarId().equals(car.getCarId())) {
                    //修改该车前一个区域的占用状态为未占用;删除占用车辆信息;
                    theArea.setAreaState(0);
                    theArea.setOccupyCar(null);
                    break;
                }
            }
        }

        //决策
        if (car.getLocalTheArea().getAreaType().equals(1)) {
            //停车场决策
            parkAreaDecision(car);
        } else if (car.getLocalTheArea().getAreaType().equals(2)) {
            //装载区决策
            workAreaDecision(car);
        } else if (car.getLocalTheArea().getAreaType().equals(3)) {
            //排土场决策
            workAreaDecision(car);
        } else if (car.getLocalTheArea().getAreaType().equals(4)) {
            //卸载区决策
            workAreaDecision(car);
        } else if (car.getLocalTheArea().getAreaType().equals(5)) {
            //道路区域决策
            roadAreaDecision(car);
        } else if (car.getLocalTheArea().getAreaType().equals(6)) {
            //路口区域决策
            crossroadsAreaDecision(car);
        } else {
            throw new RuntimeException("车辆所在道路区域异常");
        }

        car.setLocalPath(localPath);
        result.setCar(car);
        result.setTheAreas(theAreas);
        return result;
    }


    /**
     * 当前车辆所在区域为路口决策
     *
     * @param car 车辆详情
     */
    private void crossroadsAreaDecision(Car car) {
        //车辆在道路上根据后一个区域信息做决策
        byLastAreaTypeDecision(car);
    }

    /**
     * 当前车辆所在区域为道路决策
     *
     * @param car 车辆详情
     */
    private void roadAreaDecision(Car car) {
        //根据后一个区域做决策
        byLastAreaTypeDecision(car);
    }

    /**
     * 当前车辆所在区域为作业区域决策
     *
     * @param car 车辆详情
     */
    private void workAreaDecision(Car car) {
        TheArea localTheArea = car.getLocalTheArea();
        if (car.getCarTask() == 0) {
            //当前车辆在作业区域并且状态为待机
            workAreaStandbyDecision(car);
        } else if (car.getCarTask() == 1) {
            //当前车辆在作业区域并且状态为行驶
            workAreaDrivingDecision(car);
        } else if (car.getCarTask() == 2) {
            //当前车辆在作业区域并且状态为装载
            //判断区域类型是不是装载区
            if (localTheArea.getAreaType() == 2) {
                if (car.getCarLoadState() == 0) {
                    //轻车,正在装载,修改装载状态为重车
                    car.setCarLoadState(1);
                } else {
                    //重车,装载结束,修改状态为待机,等待下一次作业规划
                    car.setCarPreTask(car.getCarTask());
                    car.setCarTask(0);
                }
            } else {
                //区域不为装载区但是状态为装载,不符合逻辑,抛出异常
                throw new RuntimeException("当前车辆: " + car.getCarId() + " 在区域: " + localTheArea.getAreaId() + "状态错误,状态为" + car.getCarTask());
            }
        } else if (car.getCarTask() == 3) {
            //当前车辆在作业区域并且状态为排土
            if (localTheArea.getAreaType() == 3) {
                if (car.getCarLoadState() == 1) {
                    car.setCarLoadState(0);
                } else {
                    //轻车,卸载完成,待机,等待下一次作业规划
                    car.setCarPreTask(car.getCarTask());
                    car.setCarTask(0);
                }
            } else {
                //区域不为卸载区但是状态为卸载,不符合逻辑,抛出异常
                throw new RuntimeException("当前车辆: " + car.getCarId() + " 在区域: " + localTheArea.getAreaId() + "状态错误,状态为" + car.getCarTask());
            }

        } else if (car.getCarTask() == 4) {
            if (localTheArea.getAreaType() == 4) {
                //重车正在排土,修改状态为轻车,排土结束
                if (car.getCarLoadState() == 1) {
                    car.setCarLoadState(0);
                } else {
                    //轻车,排土完成,待机,等待下一次作业规划
                    car.setCarPreTask(car.getCarTask());
                    car.setCarTask(0);
                }
            } else {
                //区域不为排土场区但是状态为排土,不符合逻辑,抛出异常
                throw new RuntimeException("当前车辆: " + car.getCarId() + " 在区域: " + localTheArea.getAreaId() + "状态错误,状态为" + car.getCarTask());
            }
        }
    }

    /**
     * 作业区域待机决策
     *
     * @param car 车辆详情
     */
    private void workAreaStandbyDecision(Car car) {
        WorkTheArea localArea = (WorkTheArea) car.getLocalTheArea();
        //判断前一个状态
        Integer carPreTask = car.getCarPreTask();
        if (carPreTask == null) {
            //作业区域前一状态为null,不符合逻辑
            throw new RuntimeException("前一状态为null; 当前车辆: " + car.getCarId() + " 在区域: " + localArea.getAreaId() + " 状态错误,当前状态为" + car.getCarTask() + ", 前一状态为" + car.getCarPreTask());
        } else if (carPreTask == 0) {
            throw new RuntimeException("待机前为待机; 当前车辆: " + car.getCarId() + " 在区域: " + localArea.getAreaId() + " 状态错误,当前状态为" + car.getCarTask() + ", 前一状态为" + car.getCarPreTask());
        } else if (carPreTask == 1) {
            //当前作业为待机,前一状态为行驶,当前车在排队
            //获取排队队列
            Queue<Car> queue = localArea.getQueueInfo().getQueue();
            if (queue == null || queue.size() == 0) {
                throw new RuntimeException("待机前为行驶; 当前车辆: " + car.getCarId() + " 逻辑为排队, 但是排队队列为空");
            }
            // 判断当前车在队列中是否是第一辆车
            if (queue.traverse(0).getData() == car) {
                //判断区域是否空闲
                if (localArea.getAreaState() == 0) {
                    getLocalPathAndReturn(car, localArea.getWorkLocation());
                    car.setQueueNum(null);
                    Car car1 = queue.deQueue();
                    if (car != car1) {
                        throw new RuntimeException("队列删除车辆信息错误");
                    }
                    //区域修改为被占用,占用车辆为当前车辆
                    localArea.setAreaState(1);
                    localArea.setOccupyCar(car);
                } else {
                    //判断是否需要更新排队点
                    Location queuePoint = getQueuePoint(car, 1, localArea.getQueueLocation());
                    if (!car.getCarLocation().equals(queuePoint)) {
                        getLocalPathAndReturn(car, queuePoint);
                    }
                }
            } else {
                //不是第一辆车;从排队队列获取当前车辆排队位置
                int carIsQueueNum = -1;
                for (int i = 0; i < queue.size(); i++) {
                    if (queue.traverse(i).getData().getCarId().equals(car.getCarId())) {
                        carIsQueueNum = i;
                        if (!car.getTargetPoint().equals(car.getCarLocation())) {
                            Location queuePoint = getQueuePoint(car, i + 2, localArea.getQueueLocation());
                            car.setQueueNum(i + 2);
                            getLocalPathAndReturn(car, queuePoint);
                        }
                        break;
                    }
                }
                Location queuePoint = getQueuePoint(car, carIsQueueNum + 1, localArea.getQueueLocation());
                assert queuePoint != null;
                if (!queuePoint.equals(car.getCarLocation())) {
                    getLocalPathAndReturn(car, queuePoint);
                    car.setQueueNum(carIsQueueNum + 1);
                }
                if (carIsQueueNum == -1) {
                    //车不在队列
                    throw new RuntimeException("待机前为行驶; 当前车辆: " + car.getCarId() + " 逻辑为排队, 但是排队队列不包含该车信息");
                }
            }
        } else if (carPreTask == 2) {
            if (localArea.getAreaType() != 2) {
                throw new RuntimeException("区域类型和作业状态对应错误; 当前车辆: " + car.getCarId() + " 在区域: " + localArea.getAreaId() + " 状态错误,当前状态为" + car.getCarTask() + ", 前一状态为" + car.getCarPreTask());
            } else {
                workAreaStandbyAndPreStateIsWorkDecision(car);
            }
        } else if (carPreTask == 3) {
            if (localArea.getAreaType() != 3) {
                throw new RuntimeException("区域类型和作业状态对应错误; 当前车辆: " + car.getCarId() + " 在区域: " + localArea.getAreaId() + " 状态错误,当前状态为" + car.getCarTask() + ", 前一状态为" + car.getCarPreTask());
            } else {
                workAreaStandbyAndPreStateIsWorkDecision(car);
            }
        } else if (carPreTask == 4) {
            if (localArea.getAreaType() != 4) {
                throw new RuntimeException("区域类型和作业状态对应错误; 当前车辆: " + car.getCarId() + " 在区域: " + localArea.getAreaId() + " 状态错误,当前状态为" + car.getCarTask() + ", 前一状态为" + car.getCarPreTask());
            } else {
                workAreaStandbyAndPreStateIsWorkDecision(car);
            }
        }
    }

    /**
     * 作业区域待机前为作业决策
     *
     * @param car 车辆详情
     */
    private void workAreaStandbyAndPreStateIsWorkDecision(Car car) {
        //作业(装载,卸载,排土 --- 待机)说明在当前区域作业已经完成
        //根据有向图判断当前区域是不是有向图最后一个区域
        TheArea localTheArea = car.getLocalTheArea();

        if (car.getDirectedAreaIdList().get(car.getDirectedAreaIdList().size() - 1).equals(localTheArea.getAreaId())) {
            //当前区域为有向图最后一个区域, 需要进行下一次的作业规划,设置路径规划状态为未规划
            car.setPathPlanState(0);
            //清空全局路径,清空有向图信息
            car.setGlobalPath(null);
            car.setDirectedAreaIdList(null);
        } else if (car.getDirectedAreaIdList().get(0).equals(localTheArea.getAreaId())) {
            //当前区域为有向图第一个区域
            byLastAreaTypeDecision(car);
        } else {
            //有向图其他区域
            //作业区域在有向图为中间区域,不符合逻辑
            throw new RuntimeException("当前车辆: " + car.getCarId() + ", 所在区域为: " + localTheArea.getAreaId() + " 类型为 : " + localTheArea.getAreaType() + "但是不为首尾区域");
        }

    }

    /**
     * 作业区域行驶决策
     *
     * @param car 车辆详情
     */
    private void workAreaDrivingDecision(Car car) {
        WorkTheArea localArea = (WorkTheArea) car.getLocalTheArea();
        //判断车辆装载状态(装载区重车 || 卸载区 排土区轻车)
        if ((car.getCarLoadState() == 0 && localArea.getAreaType() == 2) || ((localArea.getAreaType() == 3 || localArea.getAreaType() == 4) && car.getCarLoadState() == 1)) {
            /*
             进了装载区车是空车(卸载区排土场重车),并且车为行驶状态的情况分析
             1. 刚进装载区(卸载区/排土场),没开始决策是否需要排队;
             2. 进入装载区(卸载区/排土场)已经开始排队,但是需要更新排队点,有排队状态的待机切换到了行驶;
             3. 已经排队结束,车辆正在前往该区域作业点;根据区域占用的车辆id和当前车辆id比较判断;
             */

            //判断当前区域是否被占用
            if (car.getLocalTheArea().getAreaState() == 1) {
                //当前区域被占用,判断占用车辆是否是当前车
                if (car.getLocalTheArea().getOccupyCar() == car) {
                    //判断是否到达目的点该区域作业点
                    if (car.getCarLocation().equals(car.getTargetPoint()) &&
                            car.getCarLocation().getX() == localArea.getWorkLocation().getX() &&
                            car.getCarLocation().getY() == localArea.getWorkLocation().getY()) {
                        //修改车辆状态
                        car.setCarPreTask(car.getCarTask());
                        //根据当前区域类型修改当前状态
                        if (localArea.getAreaType() == 2) {
                            car.setCarTask(2);
                        } else if (localArea.getAreaType() == 3) {
                            car.setCarTask(3);
                        } else if (localArea.getAreaType() == 4) {
                            car.setCarTask(4);
                        }
                    } else {
                        //没有到达终点,发送局部路径至该区域作业点
                        getLocalPathAndReturn(car, localArea.getWorkLocation());
                    }
                } else {
                    //不是被当前车占用,判断排队队列是否包含当前车辆
                    Queue<Car> queue = localArea.getQueueInfo().getQueue();
                    boolean carIsQueue = false;
                    for (int i = 0; i < queue.size(); i++) {
                        if (queue.traverse(i).getData().getCarId().equals(car.getCarId())) {
                            //当前车辆行驶状态并且在排队队列
                            //判断是否已经到达排队点
                            if (car.getCarLocation() == car.getTargetPoint()) {
                                //到达排队点,修改车辆状态为待机
                                car.setCarPreTask(car.getCarTask());
                                car.setCarTask(0);
                            } else {
                                //确认排队点的准确性,再次计算并给车辆
                                Location queuePoint = getQueuePoint(car, i + 1, localArea.getQueueLocation());
                                car.setTargetPoint(queuePoint);


                                //向排队点继续发送局部路径
                                getLocalPathAndReturn(car, car.getTargetPoint());
                            }
                            carIsQueue = true;
                            break;
                        }
                    }
                    if (!carIsQueue) {
                        //当前车信息不在排队队列,刚进作业区;计算排队点,发送局部路径到该点,将车辆信息加入队列
                        Location queuePoint = getQueuePoint(car, queue.size() + 1, localArea.getQueueLocation());
                        car.setQueueNum(queue.size() + 1);
                        getLocalPathAndReturn(car, queuePoint);
                        queue.enQueue(car);
                    }
                }
            } else {
                //当前区域没有车在作业,但是排队队列可能不为空,等待下一辆车进行调度
                //判断当前区域排队队列是否为空,只有排队队列为空,或者是第一辆车,才能直接发送路径到工作点
                Queue<Car> queue = localArea.getQueueInfo().getQueue();

                if (queue == null || queue.size() == 0) {
                    //排队队列没有其他车辆信息,发送局部路径到作业点,修改区域状态为作业
                    getLocalPathAndReturn(car, localArea.getWorkLocation());
                    localArea.setOccupyCar(car);
                    localArea.setAreaState(1);
                } else {
                    boolean carIsQueue = false;
                    int queueIndex = -1;
                    for (int i = 0; i < queue.size(); i++) {
                        if (queue.traverse(i).getData().getCarId().equals(car.getCarId())) {
                            queueIndex = i;
                            carIsQueue = true;
                            break;
                        }
                    }
                    if (queueIndex == 0) {
                        Car car1 = queue.deQueue();
                        if (!car.getCarId().equals(car1.getCarId())) {
                            throw new RuntimeException("删除队列车辆错误");
                        } else {
                            //排队结束,修改区域占用情况和区域占用车辆信息
                            localArea.setOccupyCar(car);
                            localArea.setAreaState(1);
                        }
                        getLocalPathAndReturn(car, localArea.getWorkLocation());
                        car.setQueueNum(null);

                    } else if (car.getCarLocation() == car.getTargetPoint()) {
                        car.setCarPreTask(car.getCarTask());
                        car.setCarTask(0);
                    } else {
                        getLocalPathAndReturn(car, car.getTargetPoint());
                    }
                    if (!carIsQueue) {
                        //当前车辆不在排队队列
                        Location queuePoint = getQueuePoint(car, queue.size() + 1, localArea.getWorkLocation());
                        car.setQueueNum(queue.size() + 1);
                        getLocalPathAndReturn(car, queuePoint);
                        queue.enQueue(car);
                    }
                }
            }
        } else {
            //装载区内为重车(卸载区/排土场为轻车),行驶出装载区,根据后一个区域做决策
            byLastAreaTypeDecision(car);
        }
    }

    /**
     * 当前车辆所在区域为停车场决策
     *
     * @param car 车辆详情
     */
    private void parkAreaDecision(Car car) {
        //todo 停车场,演示暂时一辆车对应一个停车位
        car.setCarTask(1);

        if (lastTheAreaInfo == null || lastTheAreaInfo.getAreaId() == null) {
            if (car.getCarLocation() == car.getGlobalPath().get(car.getGlobalPath().size() - 1)) {
                car.setGlobalPath(null);
                car.setPathPlanState(0);
                car.setCarLoadState(0);
                car.setLocalTheArea(null);
                car.setDirectedAreaIdList(null);
                car.setCarPreTask(null);
                car.setCarTask(0);
                car.setQueueNum(null);
                car.setTargetPoint(null);
                car.setLocalTheArea(null);
            } else {
                getLocalPathAndReturn(car, null);
            }
        } else if (lastTheAreaInfo.getAreaState() == 1) {
            //如果后一个区域被占用
            if (lastTheAreaInfo.getOccupyCar().equals(car)) {
                getLocalPathAndReturn(car, null);
            }
        } else {
            getLocalPathAndReturn(car, null);
            lastTheAreaInfo.setAreaState(1);
            lastTheAreaInfo.setOccupyCar(car);
        }
    }

    /**
     * 根据后一个区域类型做决策
     *
     * @param car 车辆详情
     */
    private void byLastAreaTypeDecision(Car car) {
        if (lastTheAreaInfo.getAreaType() == 1) {
            //后一个区域为停车场
            lastAreaIsParkAreaDecision(car);
        } else if (lastTheAreaInfo.getAreaType() == 2) {
            //后一个区域为装载区,因为作业区域没有缓冲区,需要进入该区域后才会做决策,当前情况直接发送局部路径即可;
//            lastAreaIsLoadAreaDecision(car);
            getLocalPathAndReturn(car, null);
        } else if (lastTheAreaInfo.getAreaType() == 3) {
            //后一个区域为排土场,因为作业区域没有缓冲区,需要进入该区域后才会做决策,当前情况直接发送局部路径即可;
//            lastAreaIsDisplacementAreaAreaDecision();
            getLocalPathAndReturn(car, null);
        } else if (lastTheAreaInfo.getAreaType() == 4) {
            //后一个区域为卸载区,因为作业区域没有缓冲区,需要进入该区域后才会做决策,当前情况直接发送局部路径即可;
//            lastAreaIsUnloadAreaDecision(car);
            getLocalPathAndReturn(car, null);
        } else if (lastTheAreaInfo.getAreaType() == 5) {
            //后一个区域为道路
            lastAreaIsRoadAreaDecision(car);
        } else if (lastTheAreaInfo.getAreaType() == 6) {
            //后一个区域为路口
            lastAreaIsCrossroadsAreaDecision(car);
        }
    }

    /**
     * 后一个区域为路口决策
     *
     * @param car 车辆详情
     */
    private void lastAreaIsCrossroadsAreaDecision(Car car) {
        //路口车辆为待机,说明车辆正在排队,获取排队队列
        CrossroadsTheArea localArea = (CrossroadsTheArea) lastTheAreaInfo;
        List<AreaQueueInfo<Car>> areaQueueInfoList = localArea.getAreaQueueInfoList();
        List<AreaBuffer> areaBufferList = localArea.getAreaBufferList();
        //获取排队队列
        Queue<Car> queueInfo = null;
        //排队队列所属区域
        for (AreaQueueInfo<Car> carAreaQueueInfo : areaQueueInfoList) {
            if (carAreaQueueInfo.getToAreaId().equals(lastTheAreaInfo.getAreaId()) && carAreaQueueInfo.getFromAreaId().equals(car.getLocalTheArea().getAreaId())) {
                queueInfo = carAreaQueueInfo.getQueue();
                break;
            }
        }


        //获取缓冲区信息
        AreaBuffer areaBuffer = null;
        for (AreaBuffer bufferInfo : areaBufferList) {
            if (bufferInfo.getToAreaId().equals(lastTheAreaInfo.getAreaId()) && bufferInfo.getFromAreaId().equals(car.getLocalTheArea().getAreaId())) {
                areaBuffer = bufferInfo;
                break;
            }
        }
        if (areaBuffer == null) {
            throw new RuntimeException("路口区域获取对应缓冲区失败  " + "由区域 :" + car.getLocalTheArea().getAreaId() + " --- " + "区域 :" + lastTheAreaInfo.getAreaId());
        }


        //根据车辆当前状态决策: (0:待机; 1:行驶; 2:装载; 3:排土; 4:卸矿)
        //路口上只有待机和行驶状态
        if (car.getCarTask() == 0) {

            //判断该车在不在队列中
            int index = -1;
            for (int i = 0; i < Objects.requireNonNull(queueInfo).size(); i++) {
                if (queueInfo.traverse(i).getData().getCarId().equals(car.getCarId())) {
                    index = i;
                }
            }

            if (index == -1) {
                throw new RuntimeException("当前车辆: " + car.getCarId() + "  状态为待机但是不在对应的排队队列");
            }
            if (index == 0) {
                //当前车在排队队列中是第一辆,判断路口区域是否空闲
                if (localArea.getAreaState() == 0) {
                    //更新队列信息: 删除队列头
                    Car car1 = queueInfo.deQueue();
                    if (car != car1) {
                        throw new RuntimeException("修改队列,删除车辆信息错误");
                    }
                    //需debug看区域状态是否改变;
                    //修改当前区域状态为占用,占用车辆为当前车辆;
                    localArea.setAreaState(1);
                    localArea.setOccupyCar(car);
                    //发送局部路径
                    getLocalPathAndReturn(car, null);
                } else if (localArea.getAreaState() == 1) {
                    //判断是否需要更新排队点
                    if (car.getTargetPoint() != areaBuffer.getQueueLocation()) {
                        car.setQueueNum(1);
                        car.setTargetPoint(areaBuffer.getQueueLocation());
                        getLocalPathAndReturn(car, car.getTargetPoint());
                    }
                }
            } else if (index < queueInfo.size()) {
                //当前车在排队队列中是不是第一辆,前面还有index辆车;
                if (car.getQueueNum() != index + 1) {
                    //当前所在排队点 != 当前所在队列中的位置,重新计算排队点
                    Location newQueuePoint = getQueuePoint(car, index + 1, areaBuffer.getQueueLocation());
                    getLocalPathAndReturn(car, newQueuePoint);
                } else {
                    int queueIndex = -1;
                    //判断当前车辆是否需要更新排队点位置
                    for (int i = 0; i < queueInfo.size(); i++) {
                        if (queueInfo.traverse(i).getData().getCarId().equals(car.getCarId())) {
                            queueIndex = i;
                            break;
                        }
                    }
                    Location queuePoint = getQueuePoint(car, queueIndex + 1, areaBuffer.getQueueLocation());
                    if (car.getTargetPoint() != null && !(car.getTargetPoint().equals(queuePoint))) {
                        car.setQueueNum(queueIndex + 1);
                        getLocalPathAndReturn(car, queuePoint);
                    }
                }
            }

        } else if (car.getCarTask() == 1) {
            //车辆为行驶
            if (car.getTargetPoint() == car.getCarLocation()) {
                //判断车辆的当前位置是否已经到达目标点,到达直接切换车辆状态为待机;
                car.setCarPreTask(car.getCarTask());
                car.setCarTask(0);
                car.setTargetPoint(null);
                return;
            }

            List<Location> bufferBorderList = areaBuffer.getBufferBorderList();
            if (RegionalOrientationAlgorithm.rayAlgorithm(car.getCarLocation(), bufferBorderList)) {
                //已经进入缓冲区
                if (queueInfo == null || queueInfo.size() == 0) {
                    //没有车排队
                    if (lastTheAreaInfo.getAreaState() == 1) {
                        //区域被占用
                        if (lastTheAreaInfo.getOccupyCar() == car) {
                            //占用车辆是当前车辆,发送局部路径
                            getLocalPathAndReturn(car, null);
                        } else {
                            //不是当前车辆,将车辆信息加入队列
                            assert queueInfo != null;
                            queueInfo.enQueue(car);
                            getLocalPathAndReturn(car, areaBuffer.getQueueLocation());
                        }
                    } else {
                        //区域空闲
                        getLocalPathAndReturn(car, null);
                        //修改区域状态占用信息
                        lastTheAreaInfo.setAreaState(1);
                        lastTheAreaInfo.setOccupyCar(car);
                    }
                } else {
                    //判断当前车是否是占用路口区域的车
                    if (lastTheAreaInfo.getOccupyCar() == car) {
                        //是的   直接发送局部路径
                        getLocalPathAndReturn(car, null);
                        return;
                    }
                    //有车排队
                    //判断当前车是否存在排队信息

                    //判断该车在不在队列中
                    if (carIsInQueue(car, queueInfo)) {
                        //当前车已经在排队队列中,但是没有到目标点,继续发送局部路径到目标点
                        getLocalPathAndReturn(car, car.getTargetPoint());
                    } else {
                        //没有在排队队列中,获取排队队列信息,计算排队点
                        int size = queueInfo.size();
                        Location queuePoint = getQueuePoint(car, size + 1, areaBuffer.getQueueLocation());
                        //发送局部路径到排队点
                        getLocalPathAndReturn(car, queuePoint);
                        queueInfo.enQueue(car);

                    }
                }
            } else {
                //没有进入缓冲区,继续发送局部路径
                getLocalPathAndReturn(car, null);
            }
        } else {
            throw new RuntimeException("当前车辆: " + car.getCarId() + " 所在路口区域: " + car.getLocalTheArea().getAreaId() + " 状态异常:  " + car.getCarTask());
        }
    }

    private boolean carIsInQueue(Car car, Queue<Car> queueInfo) {
        int queueIndex = -1;
        for (int i = 0; i < Objects.requireNonNull(queueInfo).size(); i++) {
            if (queueInfo.traverse(i).getData() == car) {
                queueIndex = i;
            }
        }
        return queueIndex != -1;
    }


    /**
     * 后一个区域为停车场: 演示版本可一辆车对应一个停车位,直接获取停车位信息,发送局部路径到停车点
     *
     * @param car 车辆详情
     */
    private void lastAreaIsParkAreaDecision(Car car) {
        //todo 一辆车对应一个停车点,获取停车点
        Location parkLocation = null;
        //发送局部路径到停车场
        getLocalPathAndReturn(car, parkLocation);
    }

    /**
     * 后一个区域为道路决策
     *
     * @param car 车辆详情
     */
    private void lastAreaIsRoadAreaDecision(Car car) {
        //直接获取局部路径并返回
        getLocalPathAndReturn(car, null);
    }

    /**
     * 获取局部路径并返回
     *
     * @param car 车辆详情
     */
    private void getLocalPathAndReturn(Car car, Location destination) {
        try {
            List<Location> locationList = new ArrayList<>();
            List<Location> globalPath = car.getGlobalPath();
            int localPoint = -1;
            int endPoint = -1;

            //todo 停车场写死,直接开始跑
            if (car.getLocalTheArea().getAreaType() == 1) {
                localPoint = 1;
            }
            if (destination == null) {

                //清空目标点信息
                car.setTargetPoint(null);
                //清空车辆排队点信息
                car.setQueueNum(null);

                //返回局部路径50个点
                for (int i = 0; i < globalPath.size(); i++) {
                    if (car.getCarLocation().getX() == globalPath.get(i).getX() && car.getCarLocation().getY() == globalPath.get(i).getY()) {
                        localPoint = i;
                        break;
                    }
                }
                assert localPoint != -1 : "当前车辆所在位置不在全局路径上";
                if (globalPath.size() - localPoint < 50) {
                    List<Location> locations = globalPath.subList(localPoint, globalPath.size());
                    locationList.addAll(locations);
                } else {
                    locationList.addAll(globalPath.subList(localPoint, localPoint + 50));
                }
            } else {
                //查找排队点的最近点,原排队点可能被稀释数据时过滤,得到的值为最新排队点
                GetNearestPoint.ReturnValue returnValue = GetNearestPoint.nearestPointAlgorithm(destination, globalPath, globalPath.get(globalPath.size() - 1));
                destination = returnValue.getLocation();
                car.setTargetPoint(destination);
                endPoint = returnValue.getIndex();
                //返回局部路径到destination点
                for (int i = 0; i < globalPath.size(); i++) {
                    if (car.getCarLocation() == globalPath.get(i)) {
                        localPoint = i;
                    }
                    if (globalPath.get(i) == destination) {
                        endPoint = i;
                    }
                    if (localPoint != -1 && endPoint != -1) {
                        break;
                    }
                }

                assert localPoint != -1 : "当前车辆所在位置不在全局路径上";
                assert endPoint != -1 : "当前车辆终点位置不在全局路径上";
                List<Location> locations;
                if (endPoint - localPoint > 50) {
                    locations = globalPath.subList(localPoint, localPoint + 50);
                } else {
                    locations = globalPath.subList(localPoint, endPoint + 1);
                }
                locationList.addAll(locations);
            }
            this.localPath = locationList;

            //判断当前车辆状态是否为行驶,不为行驶将当前状态切换到行驶;
            if (car.getCarTask() != null && car.getCarTask() != 1) {
                car.setCarPreTask(car.getCarTask());
                car.setCarTask(1);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 智能调度算法
     *
     * @param car      车辆详情
     * @param theAreas 区域信息
     * @return 目的区域id
     */
    public Integer getPurposeArea(Car car, List<TheArea> theAreas) {
        Integer returnAreaId;
        //获取目的区域逻辑,排队队列最短的区域
        //获取车辆当前所在区域
        TheArea localTheArea = car.getLocalTheArea();
        if (car.getCarLoadState() == 1) {
            //重车,去排土场或者卸载区
            if (car.getMineralTypes() == 1) {
                //装载类型为土 --- 排土场
                returnAreaId = getResultAreaIdByAreaType(localTheArea, theAreas, 3);
            } else if (car.getMineralTypes() == 2) {
                returnAreaId = getResultAreaIdByAreaType(localTheArea, theAreas, 4);
                //todo 测试
//                returnAreaId = 14;
                //装载类型为煤 --- 卸载区
            } else {
                throw new RuntimeException("当前车辆" + car.getCarId() + "没有该矿物类型");
            }
        } else {
            //轻车 --- 装载区
            returnAreaId = getResultAreaIdByAreaType(localTheArea, theAreas, 2);
        }
        return returnAreaId;
    }

    /**
     * 获取排队队列车辆最少的区域
     *
     * @param theAreas     区域信息
     * @param areaType     区域类型
     * @param localTheArea 当前区域
     * @return 区域id
     */
    private Integer getResultAreaIdByAreaType(TheArea localTheArea, List<TheArea> theAreas, int areaType) {
        List<WorkTheArea> li = new ArrayList<>();
        for (TheArea theArea : theAreas) {
            if (theArea.getAreaType() == areaType && theArea.getAreaId() != 9 && theArea.getAreaId() != 11) {
                li.add((WorkTheArea) theArea);
            }
        }
        int returnAreaSize = li.get(0).getTravelTheAreaCarList().size();
        Integer returnAreaId = Integer.parseInt(li.get(0).getAreaId().toString());
        for (WorkTheArea workTheArea : li) {
            if (returnAreaSize > workTheArea.getTravelTheAreaCarList().size()) {
                returnAreaSize = workTheArea.getTravelTheAreaCarList().size();
                returnAreaId = Integer.parseInt(workTheArea.getAreaId().toString());
            }
        }

        //判断距离
        return returnAreaId;
    }

    /**
     * 根据区域排队点,和排队位置获取排队点
     *
     * @param pointNum      排队位置>0,排在第几个
     * @param queueLocation 区域的排队点
     * @return 排队点
     */
    private Location getQueuePoint(Car car, int pointNum, Location queueLocation) {
        car.setQueueNum(pointNum);
        int resultIndex = -1;
        //计算排队点 --- 根据全局路径,找到对应的排队点,每辆车之间的间隔为多少个点,得到对应的排队点
        //获取全局路径
        List<Location> globalPath = car.getGlobalPath();
        //全局路径稀释后,排队点可能不在全局路径,查找最近点,获取新的排队点
        GetNearestPoint.ReturnValue returnValue = GetNearestPoint.nearestPointAlgorithm(queueLocation, globalPath, globalPath.get(globalPath.size() - 1));
//        for (int i = 0; i < globalPath.size(); i++) {
//            if (globalPath.get(i).equals(queueLocation)) {
//                resultIndex = i;
//            }
//        }
        resultIndex = returnValue.getIndex();
        if (car.getGlobalPath().size() > queueIntervalPointNum * pointNum) {
            try {
                return globalPath.get(resultIndex - (queueIntervalPointNum * pointNum));
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        } else {
            throw new RuntimeException("车辆" + car.getCarId() + "排队信息或者全局路径错误");
        }
    }
}


