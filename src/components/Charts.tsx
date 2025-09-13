import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, ResponsiveContainer, Legend } from 'recharts';
import { ChartData } from '../types';

interface ChartsProps {
  data: ChartData[];
}

export const Charts: React.FC<ChartsProps> = ({ data }) => {
  return (
    <div className="space-y-6">
      {/* Optimization Comparison Chart */}
      <motion.div 
        className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Before vs After Optimization</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#F3F4F6'
                }}
              />
              <Legend />
              <Bar dataKey="waitTime" fill="#EF4444" name="Original Wait Time (min)" radius={[2, 2, 0, 0]} />
              <Bar dataKey="optimizedWaitTime" fill="#10B981" name="Optimized Wait Time (min)" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Demand Prediction Chart */}
      <motion.div 
        className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Demand Prediction</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#F3F4F6'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="predicted" 
                stroke="#0EA5E9" 
                strokeWidth={3}
                name="Predicted Ridership"
                dot={{ fill: '#0EA5E9', strokeWidth: 2 }}
                activeDot={{ r: 6, stroke: '#0EA5E9', strokeWidth: 2, fill: '#1E293B' }}
              />
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke="#F59E0B" 
                strokeWidth={3}
                name="Actual Ridership"
                dot={{ fill: '#F59E0B', strokeWidth: 2 }}
                activeDot={{ r: 6, stroke: '#F59E0B', strokeWidth: 2, fill: '#1E293B' }}
                strokeDasharray="5,5"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Usage Optimization Chart */}
      <motion.div 
        className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Bus Utilization Efficiency</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#F3F4F6'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="usage" 
                stroke="#EF4444" 
                strokeWidth={3}
                name="Original Usage %"
                dot={{ fill: '#EF4444', strokeWidth: 2 }}
              />
              <Line 
                type="monotone" 
                dataKey="optimizedUsage" 
                stroke="#10B981" 
                strokeWidth={3}
                name="Optimized Usage %"
                dot={{ fill: '#10B981', strokeWidth: 2 }}
                activeDot={{ r: 6, stroke: '#10B981', strokeWidth: 2, fill: '#1E293B' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </div>
  );
};