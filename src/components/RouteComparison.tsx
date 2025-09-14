import React from "react";
import { motion } from "framer-motion";
import {
  Clock,
  Users,
  TrendingUp,
  Play,
  RotateCcw,
  SkipForward,
} from "lucide-react";
import { Route } from "../types";

interface RouteComparisonProps {
  route: Route;
}

export const RouteComparison: React.FC<RouteComparisonProps> = ({ route }) => {
  return (
    <div className="space-y-6">
      {/* Sub Navigation */}
      <div className="flex space-x-6 border-b border-gray-700 pb-4">
        <button className="text-white font-medium border-b-2 border-emerald-500 pb-2">
          Route Comparison
        </button>
        {/* <button className="text-gray-400 hover:text-gray-300 pb-2">
          Timeline View
        </button>
        <button className="text-gray-400 hover:text-gray-300 pb-2">
          Impact Analysis
        </button> */}
      </div>
      {/* Route Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">{route.name}</h2>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-full text-sm">
            <span>{route.timeSaved} saved</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-full text-sm">
            <span>{route.efficiencyGain} efficiency</span>
          </div>
        </div>
      </div>
      {/* Comparison Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Original Schedule */}
        <motion.div
          className="bg-gray-800 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center">
              <div className="w-3 h-3 bg-gray-500 rounded-full mr-3" />
              Original Schedule
            </h3>
          </div>

          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <Clock className="w-5 h-5 text-gray-400" />
              <div>
                <p className="text-2xl font-mono text-white">
                  {route.originalSchedule.startTime} -{" "}
                  {route.originalSchedule.endTime}
                </p>
                <p className="text-sm text-gray-400">
                  ({route.originalSchedule.duration})
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <Users className="w-5 h-5 text-gray-400" />
              <div>
                <p className="text-lg text-white">
                  {route.passengers} passengers
                </p>
                <p className="text-sm text-gray-400">Average capacity</p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <TrendingUp className="w-5 h-5 text-gray-400" />
              <div>
                <p className="text-lg text-white">
                  {route.originalSchedule.efficiency}% efficiency
                </p>
                <p className="text-sm text-gray-400">Performance metric</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Optimized Schedule */}
        <motion.div
          className="bg-gray-800 rounded-lg p-6 border border-emerald-500/50 relative overflow-hidden"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="absolute inset-0 bg-emerald-500/5" />

          <div className="relative z-10">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center">
                <div className="w-3 h-3 bg-emerald-500 rounded-full mr-3" />
                Optimized Schedule
              </h3>
            </div>

            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Clock className="w-5 h-5 text-emerald-400" />
                <div>
                  <p className="text-2xl font-mono text-white">
                    {route.optimizedSchedule.startTime} -{" "}
                    {route.optimizedSchedule.endTime}
                  </p>
                  <p className="text-sm text-emerald-400">
                    ({route.optimizedSchedule.duration})
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <Users className="w-5 h-5 text-emerald-400" />
                <div>
                  <p className="text-lg text-white">
                    {Math.round(route.passengers * 1.2)} passengers
                  </p>
                  <p className="text-sm text-emerald-400">Improved capacity</p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
                <div>
                  <p className="text-lg text-white">
                    {route.optimizedSchedule.efficiency}% efficiency
                  </p>
                  <p className="text-sm text-emerald-400">
                    Enhanced performance
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
      Timeline Controls
      {/* <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Time Efficiency</h3>
        </div>

        <div className="flex items-center space-x-4 mb-6">
          <motion.button
            className="p-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Play className="w-4 h-4" />
          </motion.button>

          <motion.button
            className="p-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <RotateCcw className="w-4 h-4" />
          </motion.button>

          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-400">0.5x</span>
            <span className="text-sm text-white bg-emerald-500 px-2 py-1 rounded">
              1x
            </span>
            <span className="text-sm text-gray-400">2x</span>
            <span className="text-sm text-gray-400">4x</span>
            <span className="text-sm text-gray-400">8x</span>
          </div>

          <motion.button
            className="p-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <SkipForward className="w-4 h-4" />
          </motion.button>

          <div className="text-sm text-gray-400">00:00</div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">00:00</span>
            <span className="text-gray-400">Timeline</span>
          </div>

          <div className="relative">
            <div className="w-full h-2 bg-gray-700 rounded-full">
              <motion.div
                className="h-full bg-emerald-500 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: "0%" }}
                transition={{ duration: 2 }}
              />
            </div>
            <div className="absolute top-0 left-0 w-3 h-3 bg-emerald-500 rounded-full -translate-y-0.5" />
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Current: 00:00</span>
          </div>

          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-400">Progress</p>
              <p className="text-white font-mono">0%</p>
            </div>
            <div>
              <p className="text-gray-400">Speed</p>
              <p className="text-white font-mono">1x</p>
            </div>
            <div>
              <p className="text-gray-400">Remaining</p>
              <p className="text-white font-mono">24:00</p>
            </div>
          </div>
        </div>
      </div> */}
    </div>
  );
};
