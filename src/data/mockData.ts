import { Bus, RouteStop, Alert, ScheduleEntry, ChartData, Route, Notification } from '../types';

export const initialBuses: Bus[] = [
  {
    id: 'B-001',
    x: 100,
    y: 150,
    targetX: 300,
    targetY: 150,
    occupancy: 25,
    maxCapacity: 40,
    status: 'normal',
    route: 'Route 5',
    nextStop: 'Central Station',
    speed: 2
  },
  {
    id: 'B-002',
    x: 450,
    y: 200,
    targetX: 200,
    targetY: 300,
    occupancy: 42,
    maxCapacity: 40,
    status: 'overcrowded',
    route: 'Route 12',
    nextStop: 'Mall Plaza',
    speed: 1.5
  },
  {
    id: 'B-003',
    x: 350,
    y: 350,
    targetX: 500,
    targetY: 100,
    occupancy: 15,
    maxCapacity: 40,
    status: 'delayed',
    route: 'Route 8',
    nextStop: 'University',
    speed: 1
  },
  {
    id: 'B-004',
    x: 200,
    y: 300,
    targetX: 400,
    targetY: 250,
    occupancy: 32,
    maxCapacity: 40,
    status: 'normal',
    route: 'Route 3',
    nextStop: 'Hospital',
    speed: 2.2
  }
];

export const routeStops: RouteStop[] = [
  { id: 'stop1', name: 'Central Station', x: 300, y: 150, routes: ['Route 5', 'Route 3'] },
  { id: 'stop2', name: 'Mall Plaza', x: 200, y: 300, routes: ['Route 12', 'Route 8'] },
  { id: 'stop3', name: 'University', x: 500, y: 100, routes: ['Route 8', 'Route 5'] },
  { id: 'stop4', name: 'Hospital', x: 400, y: 250, routes: ['Route 3', 'Route 12'] },
  { id: 'stop5', name: 'Airport', x: 100, y: 350, routes: ['Route 5'] },
  { id: 'stop6', name: 'Stadium', x: 450, y: 200, routes: ['Route 12', 'Route 3'] }
];

export const initialAlerts: Alert[] = [
  {
    id: '1',
    type: 'warning',
    message: 'Route 5 Delayed – Rescheduling Now...',
    timestamp: Date.now(),
    route: 'Route 5'
  }
];

export const scheduleData: ScheduleEntry[] = [
  { route: 'Route 5', stop: 'Central Station', originalTime: '08:15', optimizedTime: '08:12', improvement: -3 },
  { route: 'Route 5', stop: 'University', originalTime: '08:25', optimizedTime: '08:20', improvement: -5 },
  { route: 'Route 12', stop: 'Mall Plaza', originalTime: '08:30', optimizedTime: '08:25', improvement: -5 },
  { route: 'Route 12', stop: 'Stadium', originalTime: '08:45', optimizedTime: '08:38', improvement: -7 },
  { route: 'Route 8', stop: 'University', originalTime: '08:20', optimizedTime: '08:22', improvement: 2 },
  { route: 'Route 3', stop: 'Hospital', originalTime: '08:35', optimizedTime: '08:30', improvement: -5 }
];

export const initialChartData: ChartData[] = [
  { time: '08:00', waitTime: 12, optimizedWaitTime: 8, usage: 65, optimizedUsage: 85, predicted: 120, actual: 115 },
  { time: '08:15', waitTime: 15, optimizedWaitTime: 9, usage: 70, optimizedUsage: 88, predicted: 135, actual: 140 },
  { time: '08:30', waitTime: 18, optimizedWaitTime: 10, usage: 75, optimizedUsage: 92, predicted: 150, actual: 145 },
  { time: '08:45', waitTime: 14, optimizedWaitTime: 7, usage: 68, optimizedUsage: 90, predicted: 140, actual: 138 },
  { time: '09:00', waitTime: 11, optimizedWaitTime: 6, usage: 72, optimizedUsage: 95, predicted: 160, actual: 155 }
];

export const routes: Route[] = [
  {
    id: 'downtown-express',
    name: 'Downtown Express',
    buses: 6,
    stops: 12,
    avgWait: '8.5m',
    onTime: '92%',
    status: 'optimal',
    efficiency: 92,
    passengers: 32,
    originalSchedule: {
      startTime: '08:00',
      endTime: '08:45',
      duration: '45m',
      efficiency: 78
    },
    optimizedSchedule: {
      startTime: '08:00',
      endTime: '08:38',
      duration: '38m',
      efficiency: 92
    },
    timeSaved: '-7m',
    efficiencyGain: '+14%'
  },
  {
    id: 'university-loop',
    name: 'University Loop',
    buses: 4,
    stops: 18,
    avgWait: '12.3m',
    onTime: '87%',
    status: 'delayed',
    efficiency: 87,
    passengers: 28,
    originalSchedule: {
      startTime: '08:15',
      endTime: '09:00',
      duration: '45m',
      efficiency: 75
    },
    optimizedSchedule: {
      startTime: '08:15',
      endTime: '08:52',
      duration: '37m',
      efficiency: 87
    },
    timeSaved: '-8m',
    efficiencyGain: '+12%'
  },
  {
    id: 'airport-shuttle',
    name: 'Airport Shuttle',
    buses: 3,
    stops: 8,
    avgWait: '15.2m',
    onTime: '95%',
    status: 'active',
    efficiency: 95,
    passengers: 24,
    originalSchedule: {
      startTime: '08:30',
      endTime: '09:15',
      duration: '45m',
      efficiency: 82
    },
    optimizedSchedule: {
      startTime: '08:30',
      endTime: '09:05',
      duration: '35m',
      efficiency: 95
    },
    timeSaved: '-10m',
    efficiencyGain: '+13%'
  },
  {
    id: 'residential-circuit',
    name: 'Residential Circuit',
    buses: 8,
    stops: 24,
    avgWait: '9.8m',
    onTime: '89%',
    status: 'active',
    efficiency: 89,
    passengers: 45,
    originalSchedule: {
      startTime: '08:45',
      endTime: '09:30',
      duration: '45m',
      efficiency: 76
    },
    optimizedSchedule: {
      startTime: '08:45',
      endTime: '09:20',
      duration: '35m',
      efficiency: 89
    },
    timeSaved: '-10m',
    efficiencyGain: '+13%'
  }
];

export const notifications: Notification[] = [
  {
    id: '1',
    type: 'success',
    message: 'Route optimization completed successfully',
    timestamp: Date.now() - 30000
  },
  {
    id: '2',
    type: 'success',
    message: 'Route optimization completed successfully',
    timestamp: Date.now() - 60000
  },
  {
    id: '3',
    type: 'info',
    message: 'Route optimization completed successfully',
    timestamp: Date.now() - 90000
  }
];