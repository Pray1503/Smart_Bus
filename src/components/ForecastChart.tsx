// import React, { useState, useEffect } from "react";
// import {
//   LineChart,
//   Line,
//   XAxis,
//   YAxis,
//   CartesianGrid,
//   Tooltip,
//   Legend,
//   ResponsiveContainer,
//   AreaChart,
//   Area,
// } from "recharts";
// import axios from "axios";
// import { Button } from "./ui/button";
// import { Badge } from "./ui/badge";
// import { Card, CardContent } from "./ui/card";
// import {
//   TrendingUp,
//   Clock,
//   BarChart3,
//   RefreshCw,
//   AlertCircle,
//   CheckCircle,
//   Activity,
// } from "lucide-react";
// import { toast } from "sonner";

// const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
// const API = `${BACKEND_URL}/api`;

// export const ForecastChart = ({ selectedRoute, routes }) => {
//   const [forecastData, setForecastData] = useState([]);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [forecastHours, setForecastHours] = useState(4);
//   const [chartType, setChartType] = useState("line");

//   useEffect(() => {
//     if (selectedRoute) {
//       generateForecast();
//     }
//   }, [selectedRoute]);

//   const generateForecast = async () => {
//     if (!selectedRoute) {
//       toast.error("Please select a route first");
//       return;
//     }

//     setLoading(true);
//     setError(null);

//     try {
//       const response = await axios.post(`${API}/forecast`, {
//         route_id: selectedRoute,
//         hours_ahead: forecastHours,
//         include_weather: true,
//       });

//       const formattedData = response.data.predictions.map(
//         (prediction, index) => ({
//           time: new Date(prediction.timestamp).toLocaleTimeString("en-US", {
//             hour: "2-digit",
//             minute: "2-digit",
//           }),
//           hour: new Date(prediction.timestamp).getHours(),
//           predicted: prediction.predicted_ridership,
//           lowerBound: prediction.lower_bound,
//           upperBound: prediction.upper_bound,
//           trend: prediction.trend || 0,
//           timestamp: prediction.timestamp,
//         })
//       );

//       setForecastData(formattedData);
//       toast.success("Forecast generated successfully");
//     } catch (error) {
//       console.error("Forecast error:", error);
//       setError("Failed to generate forecast");
//       toast.error("Failed to generate forecast");
//     } finally {
//       setLoading(false);
//     }
//   };

//   const getSelectedRouteName = () => {
//     const route = routes.find((r) => r.route_id === selectedRoute);
//     return route
//       ? `${route.source} → ${route.destination}`
//       : `Route ${selectedRoute}`;
//   };

//   const getMaxPrediction = () => {
//     return Math.max(...forecastData.map((d) => d.predicted), 0);
//   };

//   const getPeakHour = () => {
//     if (forecastData.length === 0) return null;
//     const peak = forecastData.reduce((max, current) =>
//       current.predicted > max.predicted ? current : max
//     );
//     return peak;
//   };

//   const getAveragePrediction = () => {
//     if (forecastData.length === 0) return 0;
//     return (
//       forecastData.reduce((sum, d) => sum + d.predicted, 0) /
//       forecastData.length
//     );
//   };

//   const CustomTooltip = ({ active, payload, label }) => {
//     if (active && payload && payload.length) {
//       return (
//         <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
//           <p className="font-medium text-gray-900">{`Time: ${label}`}</p>
//           <p className="text-blue-600">
//             {`Predicted: ${payload[0].value} passengers`}
//           </p>
//           {payload[1] && (
//             <p className="text-gray-500">
//               {`Range: ${payload[1].value} - ${payload[2]?.value || 0}`}
//             </p>
//           )}
//         </div>
//       );
//     }
//     return null;
//   };

//   return (
//     <div className="space-y-6">
//       {/* Controls */}
//       <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
//         <div>
//           <h3 className="text-lg font-semibold text-gray-900">
//             Ridership Forecast - {getSelectedRouteName()}
//           </h3>
//           <p className="text-sm text-gray-600">
//             AI-powered demand prediction for optimal resource planning
//           </p>
//         </div>

//         <div className="flex items-center space-x-2">
//           <select
//             value={forecastHours}
//             onChange={(e) => setForecastHours(parseInt(e.target.value))}
//             className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
//           >
//             <option value={2}>2 hours</option>
//             <option value={4}>4 hours</option>
//             <option value={8}>8 hours</option>
//             <option value={12}>12 hours</option>
//             <option value={24}>24 hours</option>
//           </select>

//           <select
//             value={chartType}
//             onChange={(e) => setChartType(e.target.value)}
//             className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
//           >
//             <option value="line">Line Chart</option>
//             <option value="area">Area Chart</option>
//           </select>

//           <Button
//             onClick={generateForecast}
//             disabled={loading || !selectedRoute}
//             className="flex items-center space-x-2"
//           >
//             <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
//             <span>{loading ? "Generating..." : "Update"}</span>
//           </Button>
//         </div>
//       </div>

//       {/* Forecast Metrics */}
//       {forecastData.length > 0 && (
//         <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
//           <Card>
//             <CardContent className="p-4">
//               <div className="flex items-center justify-between">
//                 <div>
//                   <p className="text-sm font-medium text-gray-600">
//                     Peak Ridership
//                   </p>
//                   <p className="text-2xl font-bold text-blue-600">
//                     {getMaxPrediction()}
//                   </p>
//                   <p className="text-xs text-gray-500">
//                     Expected at {getPeakHour()?.time}
//                   </p>
//                 </div>
//                 <TrendingUp className="h-8 w-8 text-blue-500" />
//               </div>
//             </CardContent>
//           </Card>

//           <Card>
//             <CardContent className="p-4">
//               <div className="flex items-center justify-between">
//                 <div>
//                   <p className="text-sm font-medium text-gray-600">
//                     Average Demand
//                   </p>
//                   <p className="text-2xl font-bold text-green-600">
//                     {Math.round(getAveragePrediction())}
//                   </p>
//                   <p className="text-xs text-gray-500">
//                     Next {forecastHours} hours
//                   </p>
//                 </div>
//                 <Activity className="h-8 w-8 text-green-500" />
//               </div>
//             </CardContent>
//           </Card>

//           <Card>
//             <CardContent className="p-4">
//               <div className="flex items-center justify-between">
//                 <div>
//                   <p className="text-sm font-medium text-gray-600">
//                     Forecast Quality
//                   </p>
//                   <div className="flex items-center space-x-2">
//                     <CheckCircle className="h-5 w-5 text-green-500" />
//                     <Badge variant="success" className="text-xs">
//                       High Confidence
//                     </Badge>
//                   </div>
//                   <p className="text-xs text-gray-500">
//                     Based on historical patterns
//                   </p>
//                 </div>
//                 <BarChart3 className="h-8 w-8 text-purple-500" />
//               </div>
//             </CardContent>
//           </Card>
//         </div>
//       )}

//       {/* Chart */}
//       <Card>
//         <CardContent className="p-6">
//           {error && (
//             <div className="flex items-center justify-center h-64 text-red-500">
//               <div className="text-center">
//                 <AlertCircle className="h-12 w-12 mx-auto mb-4" />
//                 <p className="text-lg font-medium">{error}</p>
//                 <Button
//                   onClick={generateForecast}
//                   className="mt-4"
//                   variant="outline"
//                 >
//                   Try Again
//                 </Button>
//               </div>
//             </div>
//           )}

//           {loading && (
//             <div className="flex items-center justify-center h-64">
//               <div className="text-center">
//                 <RefreshCw className="h-12 w-12 animate-spin text-blue-500 mx-auto mb-4" />
//                 <p className="text-lg font-medium text-gray-600">
//                   Generating Forecast...
//                 </p>
//                 <p className="text-sm text-gray-500">
//                   Using AI models to predict ridership
//                 </p>
//               </div>
//             </div>
//           )}

//           {!loading && !error && forecastData.length === 0 && (
//             <div className="flex items-center justify-center h-64 text-gray-500">
//               <div className="text-center">
//                 <BarChart3 className="h-12 w-12 mx-auto mb-4" />
//                 <p className="text-lg font-medium">No Forecast Data</p>
//                 <p className="text-sm">
//                   Select a route and click Update to generate predictions
//                 </p>
//               </div>
//             </div>
//           )}

//           {!loading && !error && forecastData.length > 0 && (
//             <div className="h-80">
//               <ResponsiveContainer width="100%" height="100%">
//                 {chartType === "area" ? (
//                   <AreaChart data={forecastData}>
//                     <CartesianGrid
//                       strokeDasharray="3 3"
//                       className="opacity-30"
//                     />
//                     <XAxis
//                       dataKey="time"
//                       tick={{ fontSize: 12 }}
//                       axisLine={{ stroke: "#e5e7eb" }}
//                     />
//                     <YAxis
//                       tick={{ fontSize: 12 }}
//                       axisLine={{ stroke: "#e5e7eb" }}
//                       label={{
//                         value: "Passengers",
//                         angle: -90,
//                         position: "insideLeft",
//                         style: { textAnchor: "middle" },
//                       }}
//                     />
//                     <Tooltip content={<CustomTooltip />} />
//                     <Legend />

//                     <Area
//                       type="monotone"
//                       dataKey="upperBound"
//                       stroke="none"
//                       fill="#3B82F6"
//                       fillOpacity={0.1}
//                       name="Upper Confidence"
//                     />
//                     <Area
//                       type="monotone"
//                       dataKey="lowerBound"
//                       stroke="none"
//                       fill="#ffffff"
//                       fillOpacity={1}
//                       name="Lower Confidence"
//                     />
//                     <Area
//                       type="monotone"
//                       dataKey="predicted"
//                       stroke="#3B82F6"
//                       strokeWidth={3}
//                       fill="#3B82F6"
//                       fillOpacity={0.3}
//                       name="Predicted Ridership"
//                     />
//                   </AreaChart>
//                 ) : (
//                   <LineChart data={forecastData}>
//                     <CartesianGrid
//                       strokeDasharray="3 3"
//                       className="opacity-30"
//                     />
//                     <XAxis
//                       dataKey="time"
//                       tick={{ fontSize: 12 }}
//                       axisLine={{ stroke: "#e5e7eb" }}
//                     />
//                     <YAxis
//                       tick={{ fontSize: 12 }}
//                       axisLine={{ stroke: "#e5e7eb" }}
//                       label={{
//                         value: "Passengers",
//                         angle: -90,
//                         position: "insideLeft",
//                         style: { textAnchor: "middle" },
//                       }}
//                     />
//                     <Tooltip content={<CustomTooltip />} />
//                     <Legend />

//                     <Line
//                       type="monotone"
//                       dataKey="predicted"
//                       stroke="#3B82F6"
//                       strokeWidth={3}
//                       dot={{ fill: "#3B82F6", strokeWidth: 2, r: 4 }}
//                       activeDot={{ r: 6, stroke: "#3B82F6", strokeWidth: 2 }}
//                       name="Predicted Ridership"
//                     />
//                     <Line
//                       type="monotone"
//                       dataKey="lowerBound"
//                       stroke="#9CA3AF"
//                       strokeWidth={1}
//                       strokeDasharray="5 5"
//                       dot={false}
//                       name="Lower Bound"
//                     />
//                     <Line
//                       type="monotone"
//                       dataKey="upperBound"
//                       stroke="#9CA3AF"
//                       strokeWidth={1}
//                       strokeDasharray="5 5"
//                       dot={false}
//                       name="Upper Bound"
//                     />
//                   </LineChart>
//                 )}
//               </ResponsiveContainer>
//             </div>
//           )}
//         </CardContent>
//       </Card>

//       {/* Insights */}
//       {forecastData.length > 0 && (
//         <Card className="bg-blue-50 border-blue-200">
//           <CardContent className="p-6">
//             <h4 className="text-lg font-semibold text-blue-900 mb-3 flex items-center">
//               <TrendingUp className="h-5 w-5 mr-2" />
//               Forecast Insights
//             </h4>

//             <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
//               <div className="space-y-2">
//                 <p className="text-blue-800">
//                   <strong>Peak Period:</strong> Expected at{" "}
//                   {getPeakHour()?.time} with {getMaxPrediction()} passengers
//                 </p>
//                 <p className="text-blue-800">
//                   <strong>Average Demand:</strong>{" "}
//                   {Math.round(getAveragePrediction())} passengers per hour
//                 </p>
//               </div>

//               <div className="space-y-2">
//                 <p className="text-blue-800">
//                   <strong>Recommendation:</strong> Consider increasing frequency
//                   during peak hours
//                 </p>
//                 <p className="text-blue-800">
//                   <strong>Confidence:</strong> High accuracy based on historical
//                   patterns
//                 </p>
//               </div>
//             </div>
//           </CardContent>
//         </Card>
//       )}
//     </div>
//   );
// };

// export default ForecastChart;

import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import axios from "axios";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Card, CardContent } from "./ui/card";
import {
  TrendingUp,
  Clock,
  BarChart3,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Activity,
} from "lucide-react";
import { toast } from "sonner";

// Define the shape of the forecast prediction data
interface Prediction {
  timestamp: string;
  predicted_ridership: number;
  lower_bound: number;
  upper_bound: number;
  trend?: number;
}

interface ForecastData {
  time: string;
  hour: number;
  predicted: number;
  lowerBound: number;
  upperBound: number;
  trend?: number;
  timestamp: string;
}

// Props interface for the ForecastChart component - using any[] for routes to avoid Route type dependency
interface ForecastChartProps {
  selectedRoute: string | null;
  routes: any[];
}

export const ForecastChart: React.FC<ForecastChartProps> = ({
  selectedRoute,
  routes,
}) => {
  const [forecastData, setForecastData] = useState<ForecastData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [forecastHours, setForecastHours] = useState<number>(4);
  const [chartType, setChartType] = useState<"line" | "area">("line");

  useEffect(() => {
    if (selectedRoute) {
      generateForecast();
    }
  }, [selectedRoute]);

  const generateForecast = async () => {
    if (!selectedRoute) {
      toast.error("Please select a route first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_BACKEND_URL}/api/forecast`,
        {
          route_id: selectedRoute,
          hours_ahead: forecastHours,
          include_weather: true,
        }
      );

      const formattedData: ForecastData[] = response.data.predictions.map(
        (prediction: Prediction, index: number) => ({
          time: new Date(prediction.timestamp).toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
          }),
          hour: new Date(prediction.timestamp).getHours(),
          predicted: prediction.predicted_ridership,
          lowerBound: prediction.lower_bound,
          upperBound: prediction.upper_bound,
          trend: prediction.trend || 0,
          timestamp: prediction.timestamp,
        })
      );

      setForecastData(formattedData);
      toast.success("Forecast generated successfully");
    } catch (error) {
      console.error("Forecast error:", error);
      setError("Failed to generate forecast");
      toast.error("Failed to generate forecast");
    } finally {
      setLoading(false);
    }
  };

  const getSelectedRouteName = (): string => {
    // Type guard for routes items - assume each route has route_id, source, destination
    const route = routes.find((r: any) => r.route_id === selectedRoute);
    return route
      ? `${route.source} → ${route.destination}`
      : `Route ${selectedRoute}`;
  };

  const getMaxPrediction = (): number => {
    return Math.max(...forecastData.map((d) => d.predicted), 0);
  };

  const getPeakHour = (): ForecastData | null => {
    if (forecastData.length === 0) return null;
    return forecastData.reduce((max, current) =>
      current.predicted > max.predicted ? current : max
    );
  };

  const getAveragePrediction = (): number => {
    if (forecastData.length === 0) return 0;
    return (
      forecastData.reduce((sum, d) => sum + d.predicted, 0) /
      forecastData.length
    );
  };

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: any[];
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900">{`Time: ${label}`}</p>
          <p className="text-blue-600">
            {`Predicted: ${payload[0].value} passengers`}
          </p>
          {payload[1] && (
            <p className="text-gray-500">
              {`Range: ${payload[1].value} - ${payload[2]?.value || 0}`}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Ridership Forecast - {getSelectedRouteName()}
          </h3>
          <p className="text-sm text-gray-600">
            AI-powered demand prediction for optimal resource planning
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <select
            value={forecastHours}
            onChange={(e) => setForecastHours(parseInt(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={2}>2 hours</option>
            <option value={4}>4 hours</option>
            <option value={8}>8 hours</option>
            <option value={12}>12 hours</option>
            <option value={24}>24 hours</option>
          </select>

          <select
            value={chartType}
            onChange={(e) => setChartType(e.target.value as "line" | "area")}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="line">Line Chart</option>
            <option value="area">Area Chart</option>
          </select>

          <Button
            onClick={generateForecast}
            disabled={loading || !selectedRoute}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            <span>{loading ? "Generating..." : "Update"}</span>
          </Button>
        </div>
      </div>

      {/* Forecast Metrics */}
      {forecastData.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">
                    Peak Ridership
                  </p>
                  <p className="text-2xl font-bold text-blue-600">
                    {getMaxPrediction()}
                  </p>
                  <p className="text-xs text-gray-500">
                    Expected at {getPeakHour()?.time}
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">
                    Average Demand
                  </p>
                  <p className="text-2xl font-bold text-green-600">
                    {Math.round(getAveragePrediction())}
                  </p>
                  <p className="text-xs text-gray-500">
                    Next {forecastHours} hours
                  </p>
                </div>
                <Activity className="h-8 w-8 text-green-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">
                    Forecast Quality
                  </p>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    {/* <Badge variant="success" className="text-xs">
                      High Confidence
                    </Badge> */}
                    <Badge variant="secondary" className="text-xs">
                      High Confidence
                    </Badge>
                  </div>
                  <p className="text-xs text-gray-500">
                    Based on historical patterns
                  </p>
                </div>
                <BarChart3 className="h-8 w-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Chart */}
      <Card>
        <CardContent className="p-6">
          {error && (
            <div className="flex items-center justify-center h-64 text-red-500">
              <div className="text-center">
                <AlertCircle className="h-12 w-12 mx-auto mb-4" />
                <p className="text-lg font-medium">{error}</p>
                <Button
                  onClick={generateForecast}
                  className="mt-4"
                  variant="outline"
                >
                  Try Again
                </Button>
              </div>
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <RefreshCw className="h-12 w-12 animate-spin text-blue-500 mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-600">
                  Generating Forecast...
                </p>
                <p className="text-sm text-gray-500">
                  Using AI models to predict ridership
                </p>
              </div>
            </div>
          )}

          {!loading && !error && forecastData.length === 0 && (
            <div className="flex items-center justify-center h-64 text-gray-500">
              <div className="text-center">
                <BarChart3 className="h-12 w-12 mx-auto mb-4" />
                <p className="text-lg font-medium">No Forecast Data</p>
                <p className="text-sm">
                  Select a route and click Update to generate predictions
                </p>
              </div>
            </div>
          )}

          {!loading && !error && forecastData.length > 0 && (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                {chartType === "area" ? (
                  <AreaChart data={forecastData}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      className="opacity-30"
                    />
                    <XAxis
                      dataKey="time"
                      tick={{ fontSize: 12 }}
                      axisLine={{ stroke: "#e5e7eb" }}
                    />
                    <YAxis
                      tick={{ fontSize: 12 }}
                      axisLine={{ stroke: "#e5e7eb" }}
                      label={{
                        value: "Passengers",
                        angle: -90,
                        position: "insideLeft",
                        style: { textAnchor: "middle" },
                      }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />

                    <Area
                      type="monotone"
                      dataKey="upperBound"
                      stroke="none"
                      fill="#3B82F6"
                      fillOpacity={0.1}
                      name="Upper Confidence"
                    />
                    <Area
                      type="monotone"
                      dataKey="lowerBound"
                      stroke="none"
                      fill="#ffffff"
                      fillOpacity={1}
                      name="Lower Confidence"
                    />
                    <Area
                      type="monotone"
                      dataKey="predicted"
                      stroke="#3B82F6"
                      strokeWidth={3}
                      fill="#3B82F6"
                      fillOpacity={0.3}
                      name="Predicted Ridership"
                    />
                  </AreaChart>
                ) : (
                  <LineChart data={forecastData}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      className="opacity-30"
                    />
                    <XAxis
                      dataKey="time"
                      tick={{ fontSize: 12 }}
                      axisLine={{ stroke: "#e5e7eb" }}
                    />
                    <YAxis
                      tick={{ fontSize: 12 }}
                      axisLine={{ stroke: "#e5e7eb" }}
                      label={{
                        value: "Passengers",
                        angle: -90,
                        position: "insideLeft",
                        style: { textAnchor: "middle" },
                      }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />

                    <Line
                      type="monotone"
                      dataKey="predicted"
                      stroke="#3B82F6"
                      strokeWidth={3}
                      dot={{ fill: "#3B82F6", strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, stroke: "#3B82F6", strokeWidth: 2 }}
                      name="Predicted Ridership"
                    />
                    <Line
                      type="monotone"
                      dataKey="lowerBound"
                      stroke="#9CA3AF"
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                      name="Lower Bound"
                    />
                    <Line
                      type="monotone"
                      dataKey="upperBound"
                      stroke="#9CA3AF"
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                      name="Upper Bound"
                    />
                  </LineChart>
                )}
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Insights */}
      {forecastData.length > 0 && (
        <Card className="bg-blue-50 border-blue-200">
          <CardContent className="p-6">
            <h4 className="text-lg font-semibold text-blue-900 mb-3 flex items-center">
              <TrendingUp className="h-5 w-5 mr-2" />
              Forecast Insights
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-2">
                <p className="text-blue-800">
                  <strong>Peak Period:</strong> Expected at{" "}
                  {getPeakHour()?.time} with {getMaxPrediction()} passengers
                </p>
                <p className="text-blue-800">
                  <strong>Average Demand:</strong>{" "}
                  {Math.round(getAveragePrediction())} passengers per hour
                </p>
              </div>

              <div className="space-y-2">
                <p className="text-blue-800">
                  <strong>Recommendation:</strong> Consider increasing frequency
                  during peak hours
                </p>
                <p className="text-blue-800">
                  <strong>Confidence:</strong> High accuracy based on historical
                  patterns
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
