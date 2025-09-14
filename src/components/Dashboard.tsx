// import React, { useState, useEffect } from "react";
// import { Sidebar } from "./Sidebar";
// import { Header } from "./Header";
// import { TabNavigation } from "./TabNavigation";
// import { RouteComparison } from "./RouteComparison";
// import { NotificationPanel } from "./NotificationPanel";
// import { Charts } from "./Charts";
// import { SchedulePanel } from "./SchedulePanel";
// import { ForecastChart } from "./ForecastChart.js";
// import {
//   routes,
//   notifications as initialNotifications,
//   initialChartData,
//   scheduleData,
// } from "../data/mockData";
// import { Route, Notification, ChartData } from "../types";

// export const Dashboard: React.FC = () => {
//   const [selectedRoute, setSelectedRoute] = useState<string | null>(
//     "downtown-express"
//   );
//   const [activeTab, setActiveTab] = useState("overview");
//   const [searchTerm, setSearchTerm] = useState("");
//   const [notifications, setNotifications] =
//     useState<Notification[]>(initialNotifications);
//   const [chartData, setChartData] = useState<ChartData[]>(initialChartData);
//   const [stats, setStats] = useState({
//     activeBuses: 21,
//     avgWaitTime: 11.7,
//     onTimePercentage: 91,
//     alerts: 3,
//   });

//   useEffect(() => {
//     // Simulate real-time updates
//     const interval = setInterval(() => {
//       setStats((prev) => ({
//         ...prev,
//         avgWaitTime: 11.7 + (Math.random() - 0.5) * 2,
//         onTimePercentage: 91 + Math.floor((Math.random() - 0.5) * 6),
//       }));

//       const newTime = new Date().toLocaleTimeString("en-US", {
//         hour12: false,
//         hour: "2-digit",
//         minute: "2-digit",
//       });

//       const newDataPoint: ChartData = {
//         time: newTime,
//         waitTime: 10 + Math.random() * 8,
//         optimizedWaitTime: 5 + Math.random() * 5,
//         usage: 60 + Math.random() * 20,
//         optimizedUsage: 80 + Math.random() * 15,
//         predicted: 120 + Math.random() * 40,
//         actual: 115 + Math.random() * 45,
//       };

//       setChartData((prev) => [...prev.slice(-4), newDataPoint]);
//     }, 5000);

//     return () => {
//       clearInterval(interval);
//     };
//   }, []);

//   const dismissNotification = (id: string) => {
//     setNotifications((prev) =>
//       prev.filter((notification) => notification.id !== id)
//     );
//   };

//   const selectedRouteData = routes.find((route) => route.id === selectedRoute);

//   const renderTabContent = () => {
//     switch (activeTab) {
//       case "overview":
//         return selectedRouteData ? (
//           <RouteComparison route={selectedRouteData} />
//         ) : null;
//       case "analytics":
//         return <Charts data={chartData} />;
//       case "schedules":
//         return <SchedulePanel scheduleData={scheduleData} />;
//       case "map":
//         return <SchedulePanel scheduleData={scheduleData} />;
//       case "forecast":
//         return <ForecastChart />;
//       case "settings":
//         return (
//           <div className="p-8 text-center">
//             <h2 className="text-2xl font-bold text-white mb-4">Settings</h2>
//             <p className="text-gray-400">Settings panel coming soon...</p>
//           </div>
//         );
//       default:
//         return null;
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gray-950 flex">
//       {/* Sidebar */}
//       <Sidebar
//         routes={routes}
//         selectedRoute={selectedRoute}
//         onRouteSelect={setSelectedRoute}
//         searchTerm={searchTerm}
//         onSearchChange={setSearchTerm}
//       />

//       {/* Main Content */}
//       <div className="flex-1 flex flex-col">
//         {/* Header */}
//         <Header stats={stats} notificationCount={notifications.length} />

//         {/* Tab Navigation */}
//         <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

//         {/* Content */}
//         <div className="flex-1 p-6 overflow-auto">{renderTabContent()}</div>
//       </div>

//       {/* Notifications */}
//       <NotificationPanel
//         notifications={notifications}
//         onDismiss={dismissNotification}
//       />
//     </div>
//   );
// };

// import React, { useState, useEffect } from "react";
// import { Sidebar } from "./Sidebar";
// import { Header } from "./Header";
// import { TabNavigation } from "./TabNavigation";
// import { RouteComparison } from "./RouteComparison";
// import { NotificationPanel } from "./NotificationPanel";
// import { Charts } from "./Charts";
// import { SchedulePanel } from "./SchedulePanel";
// import { BusMap } from "./LiveMap";
// import { ForecastChart } from "./ForecastChart"; // Corrected import (no .js)
// import {
//   routes,
//   notifications as initialNotifications,
//   initialChartData,
//   scheduleData,
//   initialBuses,
//   routeStops,
// } from "../data/mockData";
// // import { Route,  Bus, RouteStop } from "../types";
// // import React from "react";
// import { Notification, ChartData, Bus, RouteStop } from "../types";

// interface BusMapProps {
//   buses: Bus[];
//   stops: RouteStop[];
// }

// export const Dashboard: React.FC = () => {
//   const [selectedRoute, setSelectedRoute] = useState<string | null>("1");
//   const [activeTab, setActiveTab] = useState("map");
//   const [searchTerm, setSearchTerm] = useState("");
//   const [notifications, setNotifications] =
//     useState<Notification[]>(initialNotifications);
//   const [chartData, setChartData] = useState<ChartData[]>(initialChartData);
//   const [stats, setStats] = useState({
//     activeBuses: initialBuses.length,
//     avgWaitTime: 11.7,
//     onTimePercentage: 91,
//     alerts: 3,
//   });

//   useEffect(() => {
//     const interval = setInterval(() => {
//       setStats((prev) => ({
//         ...prev,
//         avgWaitTime: 11.7 + (Math.random() - 0.5) * 2,
//         onTimePercentage: 91 + Math.floor((Math.random() - 0.5) * 6),
//       }));

//       const newTime = new Date().toLocaleTimeString("en-US", {
//         hour12: false,
//         hour: "2-digit",
//         minute: "2-digit",
//       });

//       const newDataPoint: ChartData = {
//         time: newTime,
//         waitTime: 10 + Math.random() * 8,
//         optimizedWaitTime: 5 + Math.random() * 5,
//         usage: 60 + Math.random() * 20,
//         optimizedUsage: 80 + Math.random() * 15,
//         predicted: 120 + Math.random() * 40,
//         actual: 115 + Math.random() * 45,
//       };

//       setChartData((prev) => [...prev.slice(-4), newDataPoint]);
//     }, 5000);

//     return () => {
//       clearInterval(interval);
//     };
//   }, []);

//   const dismissNotification = (id: string) => {
//     setNotifications((prev) =>
//       prev.filter((notification) => notification.id !== id)
//     );
//   };

//   const selectedRouteData = routes.find((route) => route.id === selectedRoute);

//   const renderTabContent = () => {
//     switch (activeTab) {
//       case "overview":
//         return selectedRouteData ? (
//           <RouteComparison route={selectedRouteData} />
//         ) : null;
//       case "analytics":
//         return <Charts data={chartData} />;
//       case "schedules":
//         return <SchedulePanel scheduleData={scheduleData} />;
//       case "map":
//         return <BusMap buses={initialBuses} stops={routeStops} />;
//       case "forecast":
//         return <ForecastChart selectedRoute={selectedRoute} routes={routes} />;
//       case "settings":
//         return (
//           <div className="p-8 text-center">
//             <h2 className="text-2xl font-bold text-white mb-4">Settings</h2>
//             <p className="text-gray-400">Settings panel coming soon...</p>
//           </div>
//         );
//       default:
//         return null;
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gray-950 flex">
//       <Sidebar
//         routes={routes}
//         selectedRoute={selectedRoute}
//         onRouteSelect={setSelectedRoute}
//         searchTerm={searchTerm}
//         onSearchChange={setSearchTerm}
//       />
//       <div className="flex-1 flex flex-col">
//         <Header stats={stats} notificationCount={notifications.length} />
//         <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
//         <div className="flex-1 p-6 overflow-auto">{renderTabContent()}</div>
//       </div>
//       <NotificationPanel
//         notifications={notifications}
//         onDismiss={dismissNotification}
//       />
//     </div>
//   );
// };

import React, { useState, useEffect } from "react";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { TabNavigation } from "./TabNavigation";
import { RouteComparison } from "./RouteComparison";
import { NotificationPanel } from "./NotificationPanel";
import { Charts } from "./Charts";
import { SchedulePanel } from "./SchedulePanel";
import { BusMap } from "./LiveMap";
import { ForecastChart } from "./ForecastChart.tsx"; // Corrected import (no .js)
import {
  routes,
  notifications as initialNotifications,
  initialChartData,
  scheduleData,
  initialBuses,
  routeStops,
} from "../data/mockData";
import { Notification, ChartData, Bus, Route } from "../types";

// Remove unused BusMapProps interface since BusMap doesn't use it currently
// interface BusMapProps {
//   buses: Bus[];
//   stops: RouteStop[];
// }

export const Dashboard: React.FC = () => {
  const [selectedRoute, setSelectedRoute] = useState<string | null>("1");
  const [activeTab, setActiveTab] = useState("overview");
  const [searchTerm, setSearchTerm] = useState("");
  const [notifications, setNotifications] =
    useState<Notification[]>(initialNotifications);
  const [chartData, setChartData] = useState<ChartData[]>(initialChartData);
  const [stats, setStats] = useState({
    activeBuses: initialBuses.length,
    avgWaitTime: 11.7,
    onTimePercentage: 91,
    alerts: 3,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setStats((prev) => ({
        ...prev,
        avgWaitTime: 11.7 + (Math.random() - 0.5) * 2,
        onTimePercentage: 91 + Math.floor((Math.random() - 0.5) * 6),
      }));

      const newTime = new Date().toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
      });

      const newDataPoint: ChartData = {
        time: newTime,
        waitTime: 10 + Math.random() * 8,
        optimizedWaitTime: 5 + Math.random() * 5,
        usage: 60 + Math.random() * 20,
        optimizedUsage: 80 + Math.random() * 15,
        predicted: 120 + Math.random() * 40,
        actual: 115 + Math.random() * 45,
      };

      setChartData((prev) => [...prev.slice(-4), newDataPoint]);
    }, 5000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  const dismissNotification = (id: string) => {
    setNotifications((prev) =>
      prev.filter((notification) => notification.id !== id)
    );
  };

  const selectedRouteData = routes.find((route) => route.id === selectedRoute);

  const renderTabContent = () => {
    switch (activeTab) {
      case "overview":
        return selectedRouteData ? (
          <RouteComparison route={selectedRouteData} />
        ) : null;
      case "analytics":
        return <Charts data={chartData} />;
      case "schedules":
        return <SchedulePanel scheduleData={scheduleData} />;
      case "map":
        return <BusMap />; // Remove props since BusMap generates its own data
      case "forecast":
        return <ForecastChart selectedRoute={selectedRoute} routes={routes} />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex">
      <Sidebar
        routes={routes}
        selectedRoute={selectedRoute}
        onRouteSelect={setSelectedRoute}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
      />
      <div className="flex-1 flex flex-col">
        <Header stats={stats} notificationCount={notifications.length} />
        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
        <div className="flex-1 p-6 overflow-auto">{renderTabContent()}</div>
      </div>
      <NotificationPanel
        notifications={notifications}
        onDismiss={dismissNotification}
      />
    </div>
  );
};
