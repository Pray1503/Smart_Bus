import { Dashboard } from "./components/Dashboard";

function App() {
  return <Dashboard />;
}

export default App;
// import React, { useState } from "react";
// import { motion } from "framer-motion";
// import {
//   BarChart3,
//   Calendar,
//   Settings,
//   TrendingUp,
//   Map,
//   Bus as BusIcon,
//   MapPin,
// } from "lucide-react";

// // Define types
// interface Bus {
//   id: string;
//   route: string;
//   occupancy: number;
//   maxCapacity: number;
//   status: "normal" | "delayed" | "overcrowded";
//   x: number;
//   y: number;
// }

// interface RouteStop {
//   id: string;
//   name: string;
//   x: number;
//   y: number;
// }

// // TabNavigation component
// interface TabNavigationProps {
//   activeTab: string;
//   onTabChange: (tab: string) => void;
// }

// const TabNavigation: React.FC<TabNavigationProps> = ({
//   activeTab,
//   onTabChange,
// }) => {
//   const tabs = [
//     { id: "overview", label: "Overview", icon: BarChart3 },
//     { id: "analytics", label: "Analytics", icon: TrendingUp },
//     { id: "schedules", label: "Schedules", icon: Calendar },
//     { id: "map", label: "Map", icon: Map },
//     { id: "settings", label: "Settings", icon: Settings },
//   ];

//   return (
//     <div className="bg-gray-900 border-b border-gray-700">
//       <div className="px-6">
//         <div className="flex space-x-8">
//           {tabs.map((tab) => {
//             const Icon = tab.icon;
//             return (
//               <motion.button
//                 key={tab.id}
//                 className={`relative flex items-center space-x-2 py-4 px-2 text-sm font-medium transition-colors ${
//                   activeTab === tab.id
//                     ? "text-white"
//                     : "text-gray-400 hover:text-gray-300"
//                 }`}
//                 onClick={() => onTabChange(tab.id)}
//                 whileHover={{ scale: 1.02 }}
//                 whileTap={{ scale: 0.98 }}
//               >
//                 <Icon className="w-4 h-4" />
//                 <span>{tab.label}</span>
//                 {activeTab === tab.id && (
//                   <motion.div
//                     className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-500"
//                     layoutId="activeTab"
//                     initial={false}
//                     transition={{ type: "spring", stiffness: 500, damping: 30 }}
//                   />
//                 )}
//               </motion.button>
//             );
//           })}
//         </div>
//       </div>
//     </div>
//   );
// };

// // LiveMap component
// interface LiveMapProps {
//   buses: Bus[];
//   stops: RouteStop[];
// }

// const LiveMap: React.FC<LiveMapProps> = ({ buses, stops }) => {
//   const getStatusColor = (status: Bus["status"]) => {
//     switch (status) {
//       case "normal":
//         return "text-emerald-400";
//       case "delayed":
//         return "text-amber-400";
//       case "overcrowded":
//         return "text-red-400";
//       default:
//         return "text-gray-400";
//     }
//   };

//   const getStatusBg = (status: Bus["status"]) => {
//     switch (status) {
//       case "normal":
//         return "bg-emerald-500/20 border-emerald-400";
//       case "delayed":
//         return "bg-amber-500/20 border-amber-400";
//       case "overcrowded":
//         return "bg-red-500/20 border-red-400";
//       default:
//         return "bg-gray-500/20 border-gray-400";
//     }
//   };

//   return (
//     <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
//       <div className="flex items-center justify-between mb-4">
//         <h3 className="text-lg font-semibold text-white">
//           AMTS Live Bus Tracking
//         </h3>
//         <div className="flex items-center space-x-4 text-xs">
//           <div className="flex items-center space-x-1">
//             <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
//             <span className="text-gray-300">Normal</span>
//           </div>
//           <div className="flex items-center space-x-1">
//             <div className="w-2 h-2 bg-amber-400 rounded-full"></div>
//             <span className="text-gray-300">Delayed</span>
//           </div>
//           <div className="flex items-center space-x-1">
//             <div className="w-2 h-2 bg-red-400 rounded-full"></div>
//             <span className="text-gray-300">Overcrowded</span>
//           </div>
//         </div>
//       </div>

//       <div className="relative bg-gray-800/50 rounded-lg h-96 overflow-hidden border border-gray-600">
//         {/* Route Lines */}
//         <svg className="absolute inset-0 w-full h-full" style={{ zIndex: 1 }}>
//           {stops.map((stop, index) =>
//             stops
//               .slice(index + 1)
//               .map((nextStop) => (
//                 <line
//                   key={`${stop.id}-${nextStop.id}`}
//                   x1={stop.x}
//                   y1={stop.y}
//                   x2={nextStop.x}
//                   y2={nextStop.y}
//                   stroke="rgb(59, 130, 246)"
//                   strokeWidth="2"
//                   strokeOpacity="0.3"
//                   strokeDasharray="5,5"
//                 />
//               ))
//           )}
//         </svg>

//         {/* Bus Stops */}
//         {stops.map((stop) => (
//           <motion.div
//             key={stop.id}
//             className="absolute transform -translate-x-1/2 -translate-y-1/2"
//             style={{ left: stop.x, top: stop.y, zIndex: 2 }}
//             whileHover={{ scale: 1.1 }}
//           >
//             <div className="flex flex-col items-center">
//               <MapPin className="w-4 h-4 text-blue-400" />
//               <span className="text-xs text-gray-300 mt-1 bg-gray-900/80 px-2 py-1 rounded">
//                 {stop.name}
//               </span>
//             </div>
//           </motion.div>
//         ))}

//         {/* Buses */}
//         {buses.map((bus) => (
//           <motion.div
//             key={bus.id}
//             className="absolute transform -translate-x-1/2 -translate-y-1/2"
//             animate={{
//               x: bus.x,
//               y: bus.y,
//             }}
//             transition={{
//               duration: 2,
//               ease: "linear",
//             }}
//             style={{ zIndex: 3 }}
//           >
//             <motion.div
//               className={`relative p-2 rounded-lg border-2 ${getStatusBg(
//                 bus.status
//               )}`}
//               whileHover={{ scale: 1.1 }}
//               animate={{
//                 boxShadow:
//                   bus.status === "normal"
//                     ? "0 0 20px rgba(16, 185, 129, 0.3)"
//                     : bus.status === "delayed"
//                     ? "0 0 20px rgba(245, 158, 11, 0.3)"
//                     : "0 0 20px rgba(239, 68, 68, 0.3)",
//               }}
//             >
//               <BusIcon className={`w-6 h-6 ${getStatusColor(bus.status)}`} />
//               {/* Bus Info */}
//               <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-gray-900/90 px-2 py-1 rounded text-xs text-white whitespace-nowrap">
//                 <div className="font-semibold">{bus.id}</div>
//                 <div className="text-gray-300">
//                   {bus.occupancy}/{bus.maxCapacity}
//                 </div>
//               </div>
//               {/* Route Info */}
//               <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 bg-gray-900/90 px-2 py-1 rounded text-xs text-white whitespace-nowrap">
//                 <div className="text-blue-400">Route {bus.route}</div>
//               </div>
//             </motion.div>
//           </motion.div>
//         ))}
//       </div>
//     </div>
//   );
// };

// // Dashboard component
// const Dashboard: React.FC = () => {
//   const [activeTab, setActiveTab] = useState<string>("map");

//   // Sample data from AMTS_Bus_Routes_Fare.xlsx
//   const busRoutes = [
//     {
//       "Route No.": "1",
//       Source: "Ratan Park",
//       Destination: "Lal Darwaja",
//       "Distance Covered": 9.5,
//       "1st Shift Trip": 13,
//       "2nd Shift Trip": 14,
//       Fare: "Rs 20",
//     },
//     {
//       "Route No.": "4",
//       Source: "Lal Darwaja",
//       Destination: "Lal Darwaja",
//       "Distance Covered": 22.8,
//       "1st Shift Trip": 6,
//       "2nd Shift Trip": 6,
//       Fare: "Rs 30",
//     },
//     {
//       "Route No.": "5",
//       Source: "Lal Darwaja",
//       Destination: "Lal Darwaja",
//       "Distance Covered": 22.8,
//       "1st Shift Trip": 6,
//       "2nd Shift Trip": 7,
//       Fare: "Rs 30",
//     },
//     {
//       "Route No.": "14",
//       Source: "Lal Darwaja",
//       Destination: "Chosar Gam",
//       "Distance Covered": 18.7,
//       "1st Shift Trip": 6,
//       "2nd Shift Trip": 6,
//       Fare: "Rs 25",
//     },
//     {
//       "Route No.": "15",
//       Source: "Vivekanand Nagar",
//       Destination: "Civil Hospital",
//       "Distance Covered": 22.15,
//       "1st Shift Trip": 7,
//       "2nd Shift Trip": 8,
//       Fare: "Rs 30",
//     },
//     // Add more routes as needed
//   ];

//   // Generate bus data
//   const buses: Bus[] = busRoutes.map((route, index) => ({
//     id: `bus-${route["Route No."]}-${index}`,
//     route: route["Route No."],
//     occupancy: Math.floor(Math.random() * 60) + 10,
//     maxCapacity: 60,
//     status: ["normal", "delayed", "overcrowded"][
//       Math.floor(Math.random() * 3)
//     ] as "normal" | "delayed" | "overcrowded",
//     x: Math.random() * 800, // Random x position within 800px width
//     y: Math.random() * 384, // Random y position within 384px height (96% of 400px)
//   }));

//   // Generate stop data
//   const stops: RouteStop[] = [
//     ...new Set(busRoutes.flatMap((r) => [r.Source, r.Destination])),
//   ].map((name, index) => ({
//     id: `stop-${index}`,
//     name,
//     x: Math.random() * 800,
//     y: Math.random() * 384,
//   }));

//   // Placeholder components for other tabs
//   const Overview = () => <div className="p-6 text-white">Overview Content</div>;
//   const Analytics = () => (
//     <div className="p-6 text-white">Analytics Content</div>
//   );
//   const Schedules = () => (
//     <div className="p-6 text-white">Schedules Content</div>
//   );
//   const Settings = () => <div className="p-6 text-white">Settings Content</div>;

//   return (
//     <div className="min-h-screen bg-gray-900">
//       <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
//       <div className="p-6">
//         {activeTab === "overview" && <Overview />}
//         {activeTab === "analytics" && <Analytics />}
//         {activeTab === "schedules" && <Schedules />}
//         {activeTab === "map" && <LiveMap buses={buses} stops={stops} />}
//         {activeTab === "settings" && <Settings />}
//       </div>
//     </div>
//   );
// };

// // Main App component
// const App: React.FC = () => {
//   return <Dashboard />;
// };

// export default App;
