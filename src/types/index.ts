export interface Bus {
  id: string;
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  occupancy: number;
  maxCapacity: number;
  status: 'normal' | 'delayed' | 'overcrowded';
  route: string;
  nextStop: string;
  speed: number;
}

export interface RouteStop {
  id: string;
  name: string;
  x: number;
  y: number;
  routes: string[];
}

export interface Alert {
  id: string;
  type: 'warning' | 'info' | 'critical';
  message: string;
  timestamp: number;
  route?: string;
}

export interface ScheduleEntry {
  route: string;
  stop: string;
  originalTime: string;
  optimizedTime: string;
  improvement: number;
}

export interface ChartData {
  time: string;
  waitTime: number;
  optimizedWaitTime: number;
  usage: number;
  optimizedUsage: number;
  predicted: number;
  actual: number;
}

export interface Route {
  id: string;
  name: string;
  buses: number;
  stops: number;
  avgWait: string;
  onTime: string;
  status: 'active' | 'delayed' | 'optimal';
  efficiency: number;
  passengers: number;
  originalSchedule: {
    startTime: string;
    endTime: string;
    duration: string;
    efficiency: number;
  };
  optimizedSchedule: {
    startTime: string;
    endTime: string;
    duration: string;
    efficiency: number;
  };
  timeSaved: string;
  efficiencyGain: string;
}

export interface Notification {
  id: string;
  type: 'success' | 'warning' | 'info';
  message: string;
  timestamp: number;
}