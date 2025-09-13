import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Info, XCircle, X } from 'lucide-react';
import { Alert } from '../types';

interface AlertBannerProps {
  alerts: Alert[];
  onDismiss: (id: string) => void;
}

export const AlertBanner: React.FC<AlertBannerProps> = ({ alerts, onDismiss }) => {
  const getAlertIcon = (type: Alert['type']) => {
    switch (type) {
      case 'warning': return AlertTriangle;
      case 'info': return Info;
      case 'critical': return XCircle;
      default: return Info;
    }
  };

  const getAlertColors = (type: Alert['type']) => {
    switch (type) {
      case 'warning': return 'bg-amber-500/20 border-amber-400 text-amber-300';
      case 'info': return 'bg-blue-500/20 border-blue-400 text-blue-300';
      case 'critical': return 'bg-red-500/20 border-red-400 text-red-300';
      default: return 'bg-gray-500/20 border-gray-400 text-gray-300';
    }
  };

  return (
    <div className="space-y-2">
      <AnimatePresence mode="popLayout">
        {alerts.map((alert, index) => {
          const Icon = getAlertIcon(alert.type);
          return (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, y: -50, scale: 0.95 }}
              animate={{ 
                opacity: 1, 
                y: 0, 
                scale: 1,
                boxShadow: alert.type === 'critical' ? '0 0 30px rgba(239, 68, 68, 0.3)' :
                          alert.type === 'warning' ? '0 0 30px rgba(245, 158, 11, 0.3)' :
                          '0 0 30px rgba(59, 130, 246, 0.3)'
              }}
              exit={{ opacity: 0, y: -50, scale: 0.95 }}
              transition={{ 
                duration: 0.3,
                delay: index * 0.1
              }}
              className={`relative overflow-hidden rounded-lg border-2 backdrop-blur-sm ${getAlertColors(alert.type)}`}
            >
              {/* Animated background pulse */}
              <motion.div
                className="absolute inset-0 opacity-20"
                animate={{ 
                  background: [
                    'rgba(0, 0, 0, 0)', 
                    alert.type === 'critical' ? 'rgba(239, 68, 68, 0.1)' :
                    alert.type === 'warning' ? 'rgba(245, 158, 11, 0.1)' :
                    'rgba(59, 130, 246, 0.1)',
                    'rgba(0, 0, 0, 0)'
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              />

              <div className="relative flex items-center justify-between p-4">
                <div className="flex items-center space-x-3">
                  <motion.div
                    animate={{ rotate: alert.type === 'critical' ? [0, 5, -5, 0] : 0 }}
                    transition={{ duration: 0.5, repeat: alert.type === 'critical' ? Infinity : 0 }}
                  >
                    <Icon className="w-6 h-6" />
                  </motion.div>
                  <div>
                    <p className="font-medium">{alert.message}</p>
                    {alert.route && (
                      <p className="text-sm opacity-80 mt-1">{alert.route}</p>
                    )}
                  </div>
                </div>

                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => onDismiss(alert.id)}
                  className="p-1 hover:bg-white/10 rounded-full transition-colors"
                >
                  <X className="w-4 h-4" />
                </motion.button>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
};