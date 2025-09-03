import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Users, DollarSign, Target, Brain,
  BarChart3, PieChart as PieChartIcon, Activity, Zap
} from 'lucide-react';

import { 
  sampleData, 
  calculateStatistics, 
  groupBy, 
  calculateCorrelation,
  detectOutliers,
  calculateDataQuality,
  prepareTimeSeriesData
} from './lib/data';

import { 
  LinearRegression, 
  DecisionTree, 
  KMeans,
  evaluateRegression,
  crossValidate,
  calculateFeatureImportance
} from './lib/models';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedModel, setSelectedModel] = useState('linear');
  const [modelResults, setModelResults] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

  // Prepare data for analysis
  const data = useMemo(() => sampleData, []);
  const timeSeriesData = useMemo(() => prepareTimeSeriesData(data), [data]);
  const dataQuality = useMemo(() => calculateDataQuality(data), [data]);

  // Calculate key metrics
  const revenueStats = useMemo(() => calculateStatistics(data, 'revenue'), [data]);
  const customerStats = useMemo(() => calculateStatistics(data, 'customers'), [data]);
  const satisfactionStats = useMemo(() => calculateStatistics(data, 'satisfaction'), [data]);

  // Group data for visualizations
  const categoryData = useMemo(() => groupBy(data, 'category', 'revenue', 'sum'), [data]);
  const regionData = useMemo(() => groupBy(data, 'region', 'customers', 'sum'), [data]);
  const outliers = useMemo(() => detectOutliers(data, 'revenue'), [data]);

  // Correlation analysis
  const correlations = useMemo(() => {
    const fields = ['revenue', 'customers', 'satisfaction', 'marketShare'];
    const matrix = {};
    
    fields.forEach(field1 => {
      matrix[field1] = {};
      fields.forEach(field2 => {
        matrix[field1][field2] = calculateCorrelation(data, field1, field2);
      });
    });
    
    return matrix;
  }, [data]);

  // Train machine learning model
  const trainModel = async () => {
    setIsTraining(true);
    
    try {
      // Prepare features and target
      const X = data.map(d => [d.customers, d.satisfaction, d.marketShare]);
      const y = data.map(d => d.revenue);
      
      let model;
      switch (selectedModel) {
        case 'linear':
          model = new LinearRegression();
          break;
        case 'tree':
          model = new DecisionTree();
          break;
        default:
          model = new LinearRegression();
      }
      
      // Train model
      model.fit(X, y);
      
      // Make predictions
      const predictions = model.predict(X);
      
      // Evaluate model
      const evaluation = evaluateRegression(y, predictions);
      
      // Cross-validation
      const cvResults = crossValidate(model, X, y, 5);
      
      // Feature importance
      const featureNames = ['Customers', 'Satisfaction', 'Market Share'];
      const importance = calculateFeatureImportance(model, X, y, featureNames);
      
      setModelResults({
        model: selectedModel,
        evaluation,
        crossValidation: cvResults,
        featureImportance: importance,
        predictions: predictions.slice(0, 50) // Show first 50 predictions
      });
      
    } catch (error) {
      console.error('Model training failed:', error);
    } finally {
      setIsTraining(false);
    }
  };

  const MetricCard = ({ title, value, change, icon: Icon, color = "blue" }) => (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className={`h-4 w-4 text-${color}-600`} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change && (
          <p className="text-xs text-muted-foreground">
            <span className={change > 0 ? "text-green-600" : "text-red-600"}>
              {change > 0 ? "+" : ""}{change.toFixed(1)}%
            </span>
            {" from last month"}
          </p>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900">
            Advanced Analytics Dashboard
          </h1>
          <p className="text-lg text-gray-600">
            Interactive Data Science & Predictive Modeling Platform
          </p>
          <div className="flex justify-center space-x-2">
            <Badge variant="secondary">Real-time Analytics</Badge>
            <Badge variant="secondary">Machine Learning</Badge>
            <Badge variant="secondary">Data Visualization</Badge>
          </div>
        </div>

        {/* Data Quality Indicator */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Data Quality Score</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Overall Quality</span>
                <span className="font-semibold">{dataQuality?.qualityScore.toFixed(1)}%</span>
              </div>
              <Progress value={dataQuality?.qualityScore} className="w-full" />
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Missing Data: </span>
                  <span className="font-medium">{dataQuality?.missingPercentage.toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-gray-600">Duplicates: </span>
                  <span className="font-medium">{dataQuality?.duplicatePercentage.toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-gray-600">Total Records: </span>
                  <span className="font-medium">{dataQuality?.totalRows.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricCard
            title="Total Revenue"
            value={`$${(revenueStats?.sum || 0).toLocaleString()}`}
            change={12.5}
            icon={DollarSign}
            color="green"
          />
          <MetricCard
            title="Total Customers"
            value={(customerStats?.sum || 0).toLocaleString()}
            change={8.2}
            icon={Users}
            color="blue"
          />
          <MetricCard
            title="Avg Satisfaction"
            value={(satisfactionStats?.mean || 0).toFixed(1)}
            change={-2.1}
            icon={Target}
            color="yellow"
          />
          <MetricCard
            title="Outliers Detected"
            value={outliers.length}
            change={null}
            icon={TrendingUp}
            color="red"
          />
        </div>

        {/* Main Dashboard Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="modeling">ML Models</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Revenue Trend */}
              <Card>
                <CardHeader>
                  <CardTitle>Revenue Trend Over Time</CardTitle>
                  <CardDescription>Daily revenue performance</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={timeSeriesData.slice(0, 100)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="formattedDate" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="revenue" stroke="#8884d8" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Category Performance */}
              <Card>
                <CardHeader>
                  <CardTitle>Revenue by Category</CardTitle>
                  <CardDescription>Performance across business categories</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={categoryData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Regional Distribution */}
              <Card>
                <CardHeader>
                  <CardTitle>Customer Distribution by Region</CardTitle>
                  <CardDescription>Geographic customer spread</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={regionData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {regionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Satisfaction vs Revenue Scatter */}
              <Card>
                <CardHeader>
                  <CardTitle>Satisfaction vs Revenue</CardTitle>
                  <CardDescription>Relationship between customer satisfaction and revenue</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart data={data.slice(0, 100)}>
                      <CartesianGrid />
                      <XAxis dataKey="satisfaction" name="Satisfaction" />
                      <YAxis dataKey="revenue" name="Revenue" />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Data Points" data={data.slice(0, 100)} fill="#8884d8" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Statistical Summary */}
              <Card>
                <CardHeader>
                  <CardTitle>Statistical Summary</CardTitle>
                  <CardDescription>Key statistical measures</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      { label: 'Revenue', stats: revenueStats },
                      { label: 'Customers', stats: customerStats },
                      { label: 'Satisfaction', stats: satisfactionStats }
                    ].map(({ label, stats }) => (
                      <div key={label} className="border rounded p-3">
                        <h4 className="font-semibold mb-2">{label}</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>Mean: {stats?.mean.toFixed(2)}</div>
                          <div>Median: {stats?.median.toFixed(2)}</div>
                          <div>Std Dev: {stats?.stdDev.toFixed(2)}</div>
                          <div>Range: {(stats?.max - stats?.min).toFixed(2)}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Correlation Matrix */}
              <Card>
                <CardHeader>
                  <CardTitle>Correlation Matrix</CardTitle>
                  <CardDescription>Feature correlations</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {Object.keys(correlations).map(field1 => (
                      <div key={field1} className="flex space-x-2">
                        <div className="w-20 text-sm font-medium">{field1}</div>
                        {Object.keys(correlations[field1]).map(field2 => (
                          <div
                            key={field2}
                            className="w-16 h-8 flex items-center justify-center text-xs rounded"
                            style={{
                              backgroundColor: `rgba(${correlations[field1][field2] > 0 ? '34, 197, 94' : '239, 68, 68'}, ${Math.abs(correlations[field1][field2])})`
                            }}
                          >
                            {correlations[field1][field2].toFixed(2)}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ML Models Tab */}
          <TabsContent value="modeling" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Model Training */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Brain className="h-5 w-5" />
                    <span>Predictive Modeling</span>
                  </CardTitle>
                  <CardDescription>Train and evaluate machine learning models</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Select Model</label>
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full p-2 border rounded"
                    >
                      <option value="linear">Linear Regression</option>
                      <option value="tree">Decision Tree</option>
                    </select>
                  </div>
                  
                  <Button 
                    onClick={trainModel} 
                    disabled={isTraining}
                    className="w-full"
                  >
                    {isTraining ? 'Training...' : 'Train Model'}
                  </Button>

                  {modelResults && (
                    <div className="space-y-3">
                      <h4 className="font-semibold">Model Performance</h4>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>R² Score: {modelResults.evaluation.r2.toFixed(3)}</div>
                        <div>RMSE: {modelResults.evaluation.rmse.toFixed(2)}</div>
                        <div>MAE: {modelResults.evaluation.mae.toFixed(2)}</div>
                        <div>CV Score: {modelResults.crossValidation.mean.toFixed(3)}</div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Feature Importance */}
              {modelResults && (
                <Card>
                  <CardHeader>
                    <CardTitle>Feature Importance</CardTitle>
                    <CardDescription>Most influential features</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={modelResults.featureImportance}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="feature" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="importance" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* Insights Tab */}
          <TabsContent value="insights" className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Zap className="h-5 w-5" />
                    <span>Key Insights</span>
                  </CardTitle>
                  <CardDescription>Automated insights from data analysis</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <h4 className="font-semibold text-blue-900">Revenue Performance</h4>
                      <p className="text-blue-800">
                        Average revenue per transaction is ${revenueStats?.mean.toFixed(2)}. 
                        The highest performing category is {categoryData[0]?.name} with 
                        ${categoryData[0]?.value.toLocaleString()} total revenue.
                      </p>
                    </div>
                    
                    <div className="p-4 bg-green-50 rounded-lg">
                      <h4 className="font-semibold text-green-900">Customer Insights</h4>
                      <p className="text-green-800">
                        Total customer base: {customerStats?.sum.toLocaleString()} customers. 
                        {regionData[0]?.name} region has the highest customer concentration 
                        with {regionData[0]?.value.toLocaleString()} customers.
                      </p>
                    </div>
                    
                    <div className="p-4 bg-yellow-50 rounded-lg">
                      <h4 className="font-semibold text-yellow-900">Quality Alerts</h4>
                      <p className="text-yellow-800">
                        Data quality score: {dataQuality?.qualityScore.toFixed(1)}%. 
                        {outliers.length} outliers detected in revenue data. 
                        Missing data: {dataQuality?.missingPercentage.toFixed(1)}%.
                      </p>
                    </div>
                    
                    {modelResults && (
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <h4 className="font-semibold text-purple-900">Model Insights</h4>
                        <p className="text-purple-800">
                          {selectedModel === 'linear' ? 'Linear Regression' : 'Decision Tree'} model 
                          achieved R² score of {modelResults.evaluation.r2.toFixed(3)}. 
                          Most important feature: {modelResults.featureImportance[0]?.feature}.
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default App;
