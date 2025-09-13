import React from 'react';
import { motion } from 'framer-motion';
import { Clock, TrendingDown, TrendingUp } from 'lucide-react';
import { ScheduleEntry } from '../types';

interface SchedulePanelProps {
  scheduleData: ScheduleEntry[];
}

export const SchedulePanel: React.FC<SchedulePanelProps> = ({ scheduleData }) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Original Timetable */}
      <motion.div 
        className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Clock className="w-5 h-5 mr-2 text-red-400" />
          Original Timetable
        </h3>
        <div className="space-y-3">
          {scheduleData.map((entry, index) => (
            <motion.div
              key={`original-${index}`}
              className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg border border-gray-600"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
            >
              <div>
                <p className="font-medium text-white">{entry.route}</p>
                <p className="text-sm text-gray-400">{entry.stop}</p>
              </div>
              <div className="text-right">
                <p className="font-mono text-lg text-red-400">{entry.originalTime}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Optimized Timetable */}
      <motion.div 
        className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Clock className="w-5 h-5 mr-2 text-emerald-400" />
          Optimized Timetable
        </h3>
        <div className="space-y-3">
          {scheduleData.map((entry, index) => (
            <motion.div
              key={`optimized-${index}`}
              className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg border border-gray-600 relative overflow-hidden"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
            >
              {/* Improvement highlight */}
              {entry.improvement < 0 && (
                <motion.div
                  className="absolute inset-0 bg-emerald-500/10"
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: 0.5 + 0.1 * index, duration: 0.5 }}
                  style={{ transformOrigin: 'left' }}
                />
              )}
              
              <div className="relative z-10">
                <p className="font-medium text-white">{entry.route}</p>
                <p className="text-sm text-gray-400">{entry.stop}</p>
              </div>
              <div className="text-right relative z-10">
                <div className="flex items-center space-x-2">
                  <p className="font-mono text-lg text-emerald-400">{entry.optimizedTime}</p>
                  <motion.div
                    className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs ${
                      entry.improvement < 0 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'bg-red-500/20 text-red-400'
                    }`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.7 + 0.1 * index }}
                  >
                    {entry.improvement < 0 ? (
                      <TrendingDown className="w-3 h-3" />
                    ) : (
                      <TrendingUp className="w-3 h-3" />
                    )}
                    <span>{Math.abs(entry.improvement)}min</span>
                  </motion.div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};