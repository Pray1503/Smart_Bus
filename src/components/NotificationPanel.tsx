import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertTriangle, Info, X } from 'lucide-react';
import { Notification } from '../types';

interface NotificationPanelProps {
  notifications: Notification[];
  onDismiss: (id: string) => void;
}

export const NotificationPanel: React.FC<NotificationPanelProps> = ({ notifications, onDismiss }) => {
  const getNotificationIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'info': return Info;
      default: return Info;
    }
  };

  const getNotificationColors = (type: Notification['type']) => {
    switch (type) {
      case 'success': return 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400';
      case 'warning': return 'bg-amber-500/10 border-amber-500/20 text-amber-400';
      case 'info': return 'bg-blue-500/10 border-blue-500/20 text-blue-400';
      default: return 'bg-gray-500/10 border-gray-500/20 text-gray-400';
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 w-80">
      <AnimatePresence mode="popLayout">
        {notifications.map((notification) => {
          const Icon = getNotificationIcon(notification.type);
          return (
            <motion.div
              key={notification.id}
              initial={{ opacity: 0, x: 300, scale: 0.95 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 300, scale: 0.95 }}
              transition={{ duration: 0.3 }}
              className={`relative overflow-hidden rounded-lg border backdrop-blur-sm ${getNotificationColors(notification.type)}`}
            >
              <div className="flex items-center justify-between p-4">
                <div className="flex items-center space-x-3">
                  <Icon className="w-5 h-5" />
                  <p className="font-medium">{notification.message}</p>
                </div>

                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => onDismiss(notification.id)}
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