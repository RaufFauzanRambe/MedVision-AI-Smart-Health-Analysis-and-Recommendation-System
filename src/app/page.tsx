'use client'

import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion'
import { useRef, useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import { 
  Activity, 
  Brain, 
  Heart, 
  Shield, 
  Zap, 
  Users, 
  CheckCircle,
  ArrowRight,
  Stethoscope,
  Pill,
  BarChart3,
  TrendingUp,
  AlertTriangle,
  Sparkles,
  Menu,
  X,
  Play,
  Star,
  ChevronRight,
  HeartPulse,
  Scan,
  FileText,
  Bell,
  Lock,
  Globe,
  Smartphone,
  Check,
  XCircle,
  Eye,
  EyeOff,
  User,
  Mail,
  Loader2,
  RefreshCw,
  Wifi,
  WifiOff,
  Crown,
  Rocket,
  Building2
} from 'lucide-react'
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
}

const fadeIn = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.8 } }
}

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
}

const scaleIn = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.5 } }
}

// Pricing Plans Data
const pricingPlans = [
  {
    name: 'Basic',
    icon: User,
    price: { monthly: 0, yearly: 0 },
    description: 'Perfect for getting started with health monitoring',
    features: [
      { text: 'Basic symptom checker', included: true },
      { text: '5 health analyses per month', included: true },
      { text: 'Email support', included: true },
      { text: 'Health tips newsletter', included: true },
      { text: 'Advanced AI diagnostics', included: false },
      { text: 'Real-time monitoring', included: false },
      { text: 'Priority support', included: false },
      { text: 'API access', included: false },
    ],
    cta: 'Get Started Free',
    popular: false,
    gradient: 'from-slate-500 to-slate-600'
  },
  {
    name: 'Pro',
    icon: Crown,
    price: { monthly: 19, yearly: 15 },
    description: 'For individuals who want comprehensive health insights',
    features: [
      { text: 'Unlimited symptom analysis', included: true },
      { text: 'Advanced AI diagnostics', included: true },
      { text: 'Real-time health monitoring', included: true },
      { text: 'Detailed health reports', included: true },
      { text: 'Priority email support', included: true },
      { text: 'Export to PDF', included: true },
      { text: 'Family account (3 members)', included: false },
      { text: 'API access', included: false },
    ],
    cta: 'Start Pro Trial',
    popular: true,
    gradient: 'from-cyan-500 to-purple-500'
  },
  {
    name: 'Enterprise',
    icon: Building2,
    price: { monthly: 99, yearly: 79 },
    description: 'For healthcare providers and organizations',
    features: [
      { text: 'Everything in Pro', included: true },
      { text: 'Unlimited family members', included: true },
      { text: 'API access & integrations', included: true },
      { text: 'Custom branding', included: true },
      { text: 'Dedicated account manager', included: true },
      { text: 'HIPAA compliance', included: true },
      { text: '24/7 phone support', included: true },
      { text: 'Custom AI model training', included: true },
    ],
    cta: 'Contact Sales',
    popular: false,
    gradient: 'from-amber-500 to-orange-500'
  }
]

// Sample health data for chart
const generateHealthData = () => [
  { month: 'Jan', heartRate: 72 + Math.floor(Math.random() * 10) - 5, bloodPressure: 120 + Math.floor(Math.random() * 10) - 5, glucose: 95 + Math.floor(Math.random() * 10) - 5 },
  { month: 'Feb', heartRate: 74 + Math.floor(Math.random() * 10) - 5, bloodPressure: 118 + Math.floor(Math.random() * 10) - 5, glucose: 92 + Math.floor(Math.random() * 10) - 5 },
  { month: 'Mar', heartRate: 70 + Math.floor(Math.random() * 10) - 5, bloodPressure: 122 + Math.floor(Math.random() * 10) - 5, glucose: 98 + Math.floor(Math.random() * 10) - 5 },
  { month: 'Apr', heartRate: 73 + Math.floor(Math.random() * 10) - 5, bloodPressure: 119 + Math.floor(Math.random() * 10) - 5, glucose: 94 + Math.floor(Math.random() * 10) - 5 },
  { month: 'May', heartRate: 71 + Math.floor(Math.random() * 10) - 5, bloodPressure: 121 + Math.floor(Math.random() * 10) - 5, glucose: 96 + Math.floor(Math.random() * 10) - 5 },
  { month: 'Jun', heartRate: 69 + Math.floor(Math.random() * 10) - 5, bloodPressure: 117 + Math.floor(Math.random() * 10) - 5, glucose: 91 + Math.floor(Math.random() * 10) - 5 },
]

const features = [
  {
    icon: Brain,
    title: 'AI Symptom Analysis',
    description: 'Advanced machine learning algorithms analyze your symptoms and provide accurate preliminary assessments.',
    color: 'from-purple-500 to-pink-500'
  },
  {
    icon: Shield,
    title: 'Risk Assessment',
    description: 'Comprehensive health risk evaluation based on your vitals, lifestyle, and medical history.',
    color: 'from-emerald-500 to-teal-500'
  },
  {
    icon: HeartPulse,
    title: 'Real-time Monitoring',
    description: 'Track your vital signs continuously with smart alerts and trend analysis.',
    color: 'from-red-500 to-orange-500'
  },
  {
    icon: Pill,
    title: 'Smart Recommendations',
    description: 'Personalized health recommendations powered by medical knowledge base.',
    color: 'from-blue-500 to-cyan-500'
  },
  {
    icon: Scan,
    title: 'Diagnostic Support',
    description: 'AI-assisted diagnostic suggestions with confidence scores and recommended tests.',
    color: 'from-amber-500 to-yellow-500'
  },
  {
    icon: FileText,
    title: 'Health Reports',
    description: 'Generate comprehensive health reports for medical consultations.',
    color: 'from-indigo-500 to-violet-500'
  }
]

const stats = [
  { label: 'Conditions Analyzed', value: '500+', icon: Activity },
  { label: 'Symptoms Recognized', value: '1000+', icon: Stethoscope },
  { label: 'Active Users', value: '50K+', icon: Users },
  { label: 'Accuracy Rate', value: '94%', icon: TrendingUp },
]

const testimonials = [
  {
    name: 'Dr. Sarah Chen',
    role: 'Cardiologist',
    content: 'MedVision has revolutionized how I approach preliminary patient screening. The AI accuracy is remarkable.',
    rating: 5
  },
  {
    name: 'Michael Roberts',
    role: 'Patient',
    content: 'Finally, a health app that gives me clear insights about my symptoms. It helped me catch a condition early!',
    rating: 5
  },
  {
    name: 'Dr. James Wilson',
    role: 'General Practitioner',
    content: 'The comprehensive reports generated by MedVision save me valuable time during consultations.',
    rating: 5
  }
]

export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('symptoms')
  const [activeSection, setActiveSection] = useState('')
  const containerRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll()
  const y = useTransform(scrollYProgress, [0, 1], [0, -50])
  const opacity = useTransform(scrollYProgress, [0, 0.3], [1, 0])

  // Auth Modal States
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login')
  const [showPassword, setShowPassword] = useState(false)
  const [authLoading, setAuthLoading] = useState(false)
  const [authError, setAuthError] = useState('')
  const [authSuccess, setAuthSuccess] = useState('')
  const [authForm, setAuthForm] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    name: ''
  })

  // Pricing States
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'yearly'>('monthly')

  // Real-time Health Data States
  const [healthData] = useState(generateHealthData)
  const [isRealTimeActive, setIsRealTimeActive] = useState(true)
  const [lastUpdate, setLastUpdate] = useState(new Date())
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected'>('connected')
  
  // Real-time Vital Signs
  const [realTimeVitals, setRealTimeVitals] = useState({
    heartRate: 72,
    bloodPressure: { systolic: 120, diastolic: 80 },
    temperature: 36.6,
    oxygenSaturation: 98,
    glucose: 95
  })

  // Real-time update simulation
  const updateRealTimeData = useCallback(() => {
    if (!isRealTimeActive) return

    // Simulate vital signs fluctuation
    setRealTimeVitals(prev => ({
      heartRate: Math.max(60, Math.min(100, prev.heartRate + (Math.random() > 0.5 ? 1 : -1) * Math.floor(Math.random() * 3))),
      bloodPressure: {
        systolic: Math.max(100, Math.min(140, prev.bloodPressure.systolic + (Math.random() > 0.5 ? 1 : -1) * Math.floor(Math.random() * 3))),
        diastolic: Math.max(60, Math.min(90, prev.bloodPressure.diastolic + (Math.random() > 0.5 ? 1 : -1) * Math.floor(Math.random() * 2)))
      },
      temperature: Math.max(36.0, Math.min(37.5, prev.temperature + (Math.random() > 0.5 ? 0.1 : -0.1) * Math.random())),
      oxygenSaturation: Math.max(94, Math.min(100, prev.oxygenSaturation + (Math.random() > 0.5 ? 0.5 : -0.5) * Math.random())),
      glucose: Math.max(70, Math.min(140, prev.glucose + (Math.random() > 0.5 ? 2 : -2) * Math.floor(Math.random() * 3)))
    }))

    setLastUpdate(new Date())
    setConnectionStatus('connected')
  }, [isRealTimeActive])

  // Real-time interval
  useEffect(() => {
    if (isRealTimeActive) {
      const interval = setInterval(updateRealTimeData, 3000)
      return () => clearInterval(interval)
    }
  }, [isRealTimeActive, updateRealTimeData])

  // Connection status simulation
  useEffect(() => {
    const statusInterval = setInterval(() => {
      if (Math.random() > 0.95) {
        setConnectionStatus('disconnected')
        setTimeout(() => setConnectionStatus('connected'), 2000)
      }
    }, 10000)
    return () => clearInterval(statusInterval)
  }, [])

  // Smooth scroll function
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
      setActiveSection(sectionId)
    }
    setMobileMenuOpen(false)
  }

  // Auth form validation
  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(email)
  }

  const validatePassword = (password: string): { valid: boolean; message: string } => {
    if (password.length < 8) {
      return { valid: false, message: 'Password must be at least 8 characters' }
    }
    if (!/[A-Z]/.test(password)) {
      return { valid: false, message: 'Password must contain at least one uppercase letter' }
    }
    if (!/[0-9]/.test(password)) {
      return { valid: false, message: 'Password must contain at least one number' }
    }
    return { valid: true, message: '' }
  }

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthError('')
    setAuthSuccess('')
    setAuthLoading(true)

    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500))

    // Email validation
    if (!validateEmail(authForm.email)) {
      setAuthError('Please enter a valid email address')
      setAuthLoading(false)
      return
    }

    // Password validation
    const passwordValidation = validatePassword(authForm.password)
    if (!passwordValidation.valid) {
      setAuthError(passwordValidation.message)
      setAuthLoading(false)
      return
    }

    // Signup specific validation
    if (authMode === 'signup') {
      if (!authForm.name.trim()) {
        setAuthError('Please enter your name')
        setAuthLoading(false)
        return
      }
      if (authForm.password !== authForm.confirmPassword) {
        setAuthError('Passwords do not match')
        setAuthLoading(false)
        return
      }
    }

    // Simulate successful auth
    setAuthSuccess(authMode === 'login' ? 'Login successful! Redirecting...' : 'Account created successfully!')
    
    setTimeout(() => {
      setAuthModalOpen(false)
      setAuthForm({ email: '', password: '', confirmPassword: '', name: '' })
      setAuthSuccess('')
    }, 2000)
    
    setAuthLoading(false)
  }

  const openAuthModal = (mode: 'login' | 'signup') => {
    setAuthMode(mode)
    setAuthModalOpen(true)
    setAuthError('')
    setAuthSuccess('')
  }

  return (
    <div ref={containerRef} className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white overflow-x-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute top-1/3 right-1/4 w-80 h-80 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-1/4 left-1/3 w-72 h-72 bg-pink-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      {/* Navigation */}
      <motion.nav 
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.5 }}
        className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-slate-900/70 border-b border-white/10"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <motion.div 
              className="flex items-center gap-2 cursor-pointer"
              whileHover={{ scale: 1.02 }}
              onClick={() => scrollToSection('hero')}
            >
              <div className="relative">
                <img 
                  src="/logo.png" 
                  alt="MedVision Logo" 
                  className="w-10 h-10 rounded-xl object-contain"
                />
                <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-slate-900 animate-pulse ${connectionStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'}`} />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                MedVision
              </span>
            </motion.div>

            {/* Desktop Menu */}
            <div className="hidden md:flex items-center gap-8">
              {[
                { name: 'Features', id: 'features' },
                { name: 'How it Works', id: 'how-it-works' },
                { name: 'Dashboard', id: 'dashboard' },
                { name: 'Pricing', id: 'pricing' }
              ].map((item) => (
                <motion.button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className={`text-sm font-medium transition-colors relative group ${
                    activeSection === item.id ? 'text-cyan-400' : 'text-slate-300 hover:text-white'
                  }`}
                  whileHover={{ y: -2 }}
                >
                  {item.name}
                  <span className={`absolute -bottom-1 left-0 h-0.5 bg-gradient-to-r from-cyan-400 to-purple-400 transition-all duration-300 ${
                    activeSection === item.id ? 'w-full' : 'w-0 group-hover:w-full'
                  }`} />
                </motion.button>
              ))}
            </div>

            <div className="hidden md:flex items-center gap-4">
              <Button 
                variant="ghost" 
                className="text-slate-300 hover:text-white hover:bg-white/10"
                onClick={() => openAuthModal('login')}
              >
                Sign In
              </Button>
              <Button 
                className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white border-0"
                onClick={() => openAuthModal('signup')}
              >
                Get Started
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <button 
              className="md:hidden text-white"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden bg-slate-900/95 backdrop-blur-xl border-b border-white/10"
            >
              <div className="px-4 py-4 space-y-3">
                {[
                  { name: 'Features', id: 'features' },
                  { name: 'How it Works', id: 'how-it-works' },
                  { name: 'Dashboard', id: 'dashboard' },
                  { name: 'Pricing', id: 'pricing' }
                ].map((item) => (
                  <button
                    key={item.id}
                    className="block w-full text-left py-2 text-slate-300 hover:text-white"
                    onClick={() => scrollToSection(item.id)}
                  >
                    {item.name}
                  </button>
                ))}
                <div className="pt-4 flex flex-col gap-2">
                  <Button 
                    variant="outline" 
                    className="w-full border-white/20 text-white"
                    onClick={() => { setMobileMenuOpen(false); openAuthModal('login'); }}
                  >
                    Sign In
                  </Button>
                  <Button 
                    className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 text-white"
                    onClick={() => { setMobileMenuOpen(false); openAuthModal('signup'); }}
                  >
                    Get Started
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.nav>

      {/* Auth Modal */}
      <Dialog open={authModalOpen} onOpenChange={setAuthModalOpen}>
        <DialogContent className="sm:max-w-md bg-slate-900 border-white/10 text-white">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-center">
              {authMode === 'login' ? 'Welcome Back' : 'Create Account'}
            </DialogTitle>
            <DialogDescription className="text-center text-slate-400">
              {authMode === 'login' 
                ? 'Sign in to access your health dashboard' 
                : 'Join MedVision for personalized health insights'}
            </DialogDescription>
          </DialogHeader>
          
          <form onSubmit={handleAuthSubmit} className="space-y-4 mt-4">
            {authMode === 'signup' && (
              <div className="space-y-2">
                <Label htmlFor="name" className="text-slate-300">Full Name</Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                  <Input
                    id="name"
                    type="text"
                    placeholder="John Doe"
                    value={authForm.name}
                    onChange={(e) => setAuthForm({ ...authForm, name: e.target.value })}
                    className="pl-10 bg-slate-800/50 border-white/10 text-white placeholder:text-slate-500 focus:border-cyan-500"
                  />
                </div>
              </div>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="email" className="text-slate-300">Email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={authForm.email}
                  onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
                  className="pl-10 bg-slate-800/50 border-white/10 text-white placeholder:text-slate-500 focus:border-cyan-500"
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="password" className="text-slate-300">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="••••••••"
                  value={authForm.password}
                  onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
                  className="pl-10 pr-10 bg-slate-800/50 border-white/10 text-white placeholder:text-slate-500 focus:border-cyan-500"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {authForm.password && (
                <div className="text-xs space-y-1">
                  <div className={`flex items-center gap-1 ${authForm.password.length >= 8 ? 'text-green-400' : 'text-slate-500'}`}>
                    {authForm.password.length >= 8 ? <Check className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                    At least 8 characters
                  </div>
                  <div className={`flex items-center gap-1 ${/[A-Z]/.test(authForm.password) ? 'text-green-400' : 'text-slate-500'}`}>
                    {/[A-Z]/.test(authForm.password) ? <Check className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                    One uppercase letter
                  </div>
                  <div className={`flex items-center gap-1 ${/[0-9]/.test(authForm.password) ? 'text-green-400' : 'text-slate-500'}`}>
                    {/[0-9]/.test(authForm.password) ? <Check className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                    One number
                  </div>
                </div>
              )}
            </div>

            {authMode === 'signup' && (
              <div className="space-y-2">
                <Label htmlFor="confirmPassword" className="text-slate-300">Confirm Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                  <Input
                    id="confirmPassword"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="••••••••"
                    value={authForm.confirmPassword}
                    onChange={(e) => setAuthForm({ ...authForm, confirmPassword: e.target.value })}
                    className="pl-10 bg-slate-800/50 border-white/10 text-white placeholder:text-slate-500 focus:border-cyan-500"
                  />
                </div>
                {authForm.confirmPassword && authForm.password !== authForm.confirmPassword && (
                  <p className="text-xs text-red-400">Passwords do not match</p>
                )}
              </div>
            )}

            {authError && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                {authError}
              </div>
            )}

            {authSuccess && (
              <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 text-sm flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                {authSuccess}
              </div>
            )}

            <Button 
              type="submit" 
              className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white"
              disabled={authLoading}
            >
              {authLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {authMode === 'login' ? 'Signing In...' : 'Creating Account...'}
                </>
              ) : (
                authMode === 'login' ? 'Sign In' : 'Create Account'
              )}
            </Button>
          </form>

          <div className="mt-4 text-center text-sm text-slate-400">
            {authMode === 'login' ? (
              <>
                Don&apos;t have an account?{' '}
                <button 
                  onClick={() => { setAuthMode('signup'); setAuthError(''); }}
                  className="text-cyan-400 hover:underline"
                >
                  Sign up
                </button>
              </>
            ) : (
              <>
                Already have an account?{' '}
                <button 
                  onClick={() => { setAuthMode('login'); setAuthError(''); }}
                  className="text-cyan-400 hover:underline"
                >
                  Sign in
                </button>
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Hero Section */}
      <section id="hero" className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <motion.div style={{ y, opacity }} className="max-w-7xl mx-auto">
          <div className="text-center">
            <motion.div
              initial="hidden"
              animate="visible"
              variants={staggerContainer}
              className="flex justify-center mb-6"
            >
              <motion.div variants={fadeInUp}>
                <Badge variant="outline" className="px-4 py-2 border-cyan-500/50 bg-cyan-500/10 text-cyan-400">
                  <Sparkles className="w-4 h-4 mr-2" />
                  AI-Powered Health Analysis
                </Badge>
              </motion.div>
            </motion.div>

            <motion.h1
              initial="hidden"
              animate="visible"
              variants={fadeInUp}
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-6"
            >
              <span className="block">Your Personal</span>
              <span className="block bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                AI Health Assistant
              </span>
            </motion.h1>

            <motion.p
              initial="hidden"
              animate="visible"
              variants={fadeInUp}
              className="text-lg sm:text-xl text-slate-400 max-w-3xl mx-auto mb-8"
            >
              Advanced symptom analysis, risk assessment, and personalized health recommendations 
              powered by cutting-edge artificial intelligence. Take control of your health today.
            </motion.p>

            <motion.div
              initial="hidden"
              animate="visible"
              variants={fadeInUp}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Button 
                size="lg" 
                className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white px-8 h-12 text-lg border-0 group"
                onClick={() => openAuthModal('signup')}
              >
                Start Free Analysis
                <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button size="lg" variant="outline" className="border-white/20 text-white hover:bg-white/10 px-8 h-12">
                <Play className="w-5 h-5 mr-2" />
                Watch Demo
              </Button>
            </motion.div>

            {/* Trust Badges */}
            <motion.div
              initial="hidden"
              animate="visible"
              variants={fadeInUp}
              className="mt-12 flex flex-wrap items-center justify-center gap-6 text-slate-500"
            >
              <div className="flex items-center gap-2">
                <Lock className="w-4 h-4" />
                <span>HIPAA Compliant</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4" />
                <span>256-bit Encryption</span>
              </div>
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4" />
                <span>50+ Countries</span>
              </div>
            </motion.div>
          </div>

          {/* Hero Dashboard Preview - Real-time */}
          <motion.div
            initial="hidden"
            animate="visible"
            variants={scaleIn}
            className="mt-16 relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-pink-500/20 rounded-3xl blur-3xl" />
            <Card className="relative bg-slate-900/80 backdrop-blur-xl border-white/10 overflow-hidden">
              <CardHeader className="border-b border-white/10">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <CardTitle className="text-white flex items-center gap-2">
                      Health Dashboard
                      {isRealTimeActive && (
                        <Badge className="bg-green-500/20 text-green-400 border-green-500/50 animate-pulse">
                          <span className="w-2 h-2 rounded-full bg-green-400 mr-1.5 animate-pulse" />
                          Live
                        </Badge>
                      )}
                    </CardTitle>
                    <CardDescription className="text-slate-400">
                      Real-time health monitoring • Last update: {lastUpdate.toLocaleTimeString()}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      {connectionStatus === 'connected' ? (
                        <Wifi className="w-4 h-4 text-green-400" />
                      ) : (
                        <WifiOff className="w-4 h-4 text-red-400" />
                      )}
                      <span className="hidden sm:inline">{connectionStatus === 'connected' ? 'Connected' : 'Reconnecting...'}</span>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsRealTimeActive(!isRealTimeActive)}
                      className="text-slate-400 hover:text-white"
                    >
                      <RefreshCw className={`w-4 h-4 ${isRealTimeActive ? 'animate-spin' : ''}`} />
                    </Button>
                    <Bell className="w-5 h-5 text-slate-400" />
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {[
                    { label: 'Heart Rate', value: realTimeVitals.heartRate, unit: 'bpm', icon: Heart, color: 'text-red-400', trend: 'stable' },
                    { label: 'Blood Pressure', value: `${realTimeVitals.bloodPressure.systolic}/${realTimeVitals.bloodPressure.diastolic}`, unit: 'mmHg', icon: Activity, color: 'text-cyan-400', trend: 'stable' },
                    { label: 'Temperature', value: realTimeVitals.temperature.toFixed(1), unit: '°C', icon: Zap, color: 'text-amber-400', trend: 'stable' },
                    { label: 'Oxygen Level', value: realTimeVitals.oxygenSaturation.toFixed(0), unit: '%', icon: Shield, color: 'text-green-400', trend: 'stable' },
                  ].map((metric, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className="bg-slate-800/50 rounded-xl p-4 border border-white/5"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <metric.icon className={`w-5 h-5 ${metric.color}`} />
                        <motion.span 
                          key={metric.value}
                          initial={{ scale: 1.2 }}
                          animate={{ scale: 1 }}
                          className="text-xs text-green-400"
                        >
                          {metric.trend === 'stable' ? '→' : metric.trend === 'up' ? '↑' : '↓'}
                        </motion.span>
                      </div>
                      <motion.div 
                        key={metric.value}
                        initial={{ opacity: 0.5 }}
                        animate={{ opacity: 1 }}
                        className="text-2xl font-bold text-white"
                      >
                        {metric.value}
                      </motion.div>
                      <div className="text-sm text-slate-400">{metric.label}</div>
                    </motion.div>
                  ))}
                </div>

                <div className="h-64 bg-slate-800/30 rounded-xl p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={healthData}>
                      <defs>
                        <linearGradient id="colorHeart" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="month" stroke="#64748b" />
                      <YAxis stroke="#64748b" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1e293b', 
                          border: '1px solid #475569',
                          borderRadius: '8px'
                        }} 
                      />
                      <Area type="monotone" dataKey="heartRate" stroke="#ef4444" fill="url(#colorHeart)" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="grid grid-cols-2 md:grid-cols-4 gap-6"
          >
            {stats.map((stat, i) => (
              <motion.div
                key={i}
                variants={fadeInUp}
                className="text-center p-6 bg-slate-800/30 rounded-2xl border border-white/5 backdrop-blur-sm"
              >
                <stat.icon className="w-8 h-8 mx-auto mb-3 text-cyan-400" />
                <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  {stat.value}
                </div>
                <div className="text-sm text-slate-400">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeInUp}
            className="text-center mb-16"
          >
            <Badge variant="outline" className="mb-4 border-purple-500/50 bg-purple-500/10 text-purple-400">
              Features
            </Badge>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Powerful AI Health Tools
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Our comprehensive suite of health analysis tools leverages the latest in artificial intelligence 
              to provide you with accurate, actionable health insights.
            </p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {features.map((feature, i) => (
              <motion.div
                key={i}
                variants={fadeInUp}
                whileHover={{ y: -5, scale: 1.02 }}
                className="group"
              >
                <Card className="h-full bg-slate-800/30 border-white/5 hover:border-white/20 transition-all duration-300 backdrop-blur-sm">
                  <CardHeader>
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                      <feature.icon className="w-6 h-6 text-white" />
                    </div>
                    <CardTitle className="text-white group-hover:text-cyan-400 transition-colors">
                      {feature.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-slate-400">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how-it-works" className="py-20 px-4 sm:px-6 lg:px-8 bg-slate-800/20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeInUp}
            className="text-center mb-16"
          >
            <Badge variant="outline" className="mb-4 border-emerald-500/50 bg-emerald-500/10 text-emerald-400">
              How It Works
            </Badge>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Simple Steps to Better Health
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Get personalized health insights in minutes with our streamlined analysis process.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {[
              { step: 1, title: 'Enter Symptoms', desc: 'Describe your symptoms using our intuitive interface', icon: Stethoscope },
              { step: 2, title: 'AI Analysis', desc: 'Our AI processes your data against thousands of conditions', icon: Brain },
              { step: 3, title: 'Get Insights', desc: 'Receive detailed analysis with confidence scores', icon: BarChart3 },
              { step: 4, title: 'Take Action', desc: 'Follow personalized recommendations for next steps', icon: CheckCircle },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-100px" }}
                variants={fadeInUp}
                className="relative"
              >
                <div className="text-center">
                  <div className="relative inline-block mb-4">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center">
                      <item.icon className="w-8 h-8 text-white" />
                    </div>
                    <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-slate-900 border-2 border-cyan-500 flex items-center justify-center text-xs font-bold">
                      {item.step}
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{item.title}</h3>
                  <p className="text-slate-400 text-sm">{item.desc}</p>
                </div>
                {i < 3 && (
                  <div className="hidden md:block absolute top-8 left-[60%] w-[80%] h-0.5 bg-gradient-to-r from-cyan-500/50 to-transparent" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Interactive Demo Section */}
      <section id="dashboard" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeInUp}
            className="text-center mb-12"
          >
            <Badge variant="outline" className="mb-4 border-pink-500/50 bg-pink-500/10 text-pink-400">
              Interactive Demo
            </Badge>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Try MedVision Now
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Experience our AI-powered health analysis tools firsthand.
            </p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={scaleIn}
          >
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3 bg-slate-800/50 border border-white/10 mb-8">
                <TabsTrigger value="symptoms" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500 data-[state=active]:to-purple-500">
                  Symptom Checker
                </TabsTrigger>
                <TabsTrigger value="vitals" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500 data-[state=active]:to-purple-500">
                  Vital Signs
                </TabsTrigger>
                <TabsTrigger value="risks" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500 data-[state=active]:to-purple-500">
                  Risk Assessment
                </TabsTrigger>
              </TabsList>

              <TabsContent value="symptoms">
                <Card className="bg-slate-800/30 border-white/10 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      <Stethoscope className="w-5 h-5 text-cyan-400" />
                      Symptom Analysis
                    </CardTitle>
                    <CardDescription>Enter your symptoms to get AI-powered analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div className="bg-slate-700/30 rounded-xl p-4">
                          <h4 className="text-sm font-medium text-slate-300 mb-3">Common Symptoms</h4>
                          <div className="flex flex-wrap gap-2">
                            {['Headache', 'Fever', 'Cough', 'Fatigue', 'Nausea', 'Dizziness'].map((symptom) => (
                              <Badge 
                                key={symptom}
                                variant="outline"
                                className="cursor-pointer hover:bg-cyan-500/20 hover:border-cyan-500/50 transition-colors"
                              >
                                {symptom}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        
                        <div className="bg-slate-700/30 rounded-xl p-4">
                          <h4 className="text-sm font-medium text-slate-300 mb-3">Analysis Results</h4>
                          <div className="space-y-3">
                            {[
                              { condition: 'Common Cold', confidence: 78 },
                              { condition: 'Seasonal Allergies', confidence: 45 },
                              { condition: 'Sinusitis', confidence: 32 },
                            ].map((result, i) => (
                              <div key={i} className="flex items-center justify-between">
                                <span className="text-sm text-slate-400">{result.condition}</span>
                                <div className="flex items-center gap-2">
                                  <Progress value={result.confidence} className="w-20 h-2" />
                                  <span className="text-sm font-medium text-white">{result.confidence}%</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                      
                      <div className="bg-gradient-to-br from-cyan-500/10 to-purple-500/10 rounded-xl p-6 border border-white/5">
                        <AlertTriangle className="w-8 h-8 text-amber-400 mb-4" />
                        <h4 className="text-lg font-semibold text-white mb-2">Recommendation</h4>
                        <p className="text-slate-300 text-sm mb-4">
                          Based on your symptoms, we recommend resting and staying hydrated. 
                          If symptoms persist for more than 3 days, please consult a healthcare professional.
                        </p>
                        <Button className="w-full bg-gradient-to-r from-cyan-500 to-purple-500">
                          View Full Report
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="vitals">
                <Card className="bg-slate-800/30 border-white/10 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      <Activity className="w-5 h-5 text-cyan-400" />
                      Vital Signs Monitor
                    </CardTitle>
                    <CardDescription>Track and analyze your vital signs in real-time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {[
                        { label: 'Heart Rate', value: realTimeVitals.heartRate, unit: 'bpm', icon: Heart, color: 'text-red-400' },
                        { label: 'Blood Pressure', value: `${realTimeVitals.bloodPressure.systolic}/${realTimeVitals.bloodPressure.diastolic}`, unit: 'mmHg', icon: Activity, color: 'text-cyan-400' },
                        { label: 'Temperature', value: realTimeVitals.temperature.toFixed(1), unit: '°C', icon: Zap, color: 'text-amber-400' },
                        { label: 'Oxygen', value: realTimeVitals.oxygenSaturation.toFixed(0), unit: '%', icon: Shield, color: 'text-green-400' },
                      ].map((vital, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: i * 0.1 }}
                          className="bg-slate-700/30 rounded-xl p-4 text-center"
                        >
                          <vital.icon className={`w-6 h-6 mx-auto mb-2 ${vital.color}`} />
                          <motion.div 
                            key={vital.value}
                            initial={{ scale: 1.1 }}
                            animate={{ scale: 1 }}
                            className="text-2xl font-bold text-white"
                          >
                            {vital.value}
                          </motion.div>
                          <div className="text-sm text-slate-400">{vital.unit}</div>
                          <div className="text-xs text-green-400 mt-1">Normal</div>
                        </motion.div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="risks">
                <Card className="bg-slate-800/30 border-white/10 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      <Shield className="w-5 h-5 text-cyan-400" />
                      Risk Assessment
                    </CardTitle>
                    <CardDescription>Comprehensive health risk evaluation</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { category: 'Cardiovascular', score: 25, level: 'Low', color: 'bg-green-500' },
                        { category: 'Diabetes', score: 15, level: 'Minimal', color: 'bg-green-400' },
                        { category: 'Respiratory', score: 35, level: 'Low-Moderate', color: 'bg-yellow-500' },
                      ].map((risk, i) => (
                        <div key={i} className="bg-slate-700/30 rounded-xl p-4">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium text-white">{risk.category} Risk</span>
                            <Badge className={`${risk.color} text-white`}>{risk.level}</Badge>
                          </div>
                          <div className="flex items-center gap-4">
                            <Progress value={risk.score} className="flex-1 h-2" />
                            <span className="text-sm text-slate-400">{risk.score}/100</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </motion.div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-20 px-4 sm:px-6 lg:px-8 bg-slate-800/20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeInUp}
            className="text-center mb-12"
          >
            <Badge variant="outline" className="mb-4 border-amber-500/50 bg-amber-500/10 text-amber-400">
              Pricing
            </Badge>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Choose Your Health Plan
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto mb-8">
              Select the perfect plan for your health monitoring needs. All plans include our core AI analysis features.
            </p>
            
            {/* Billing Toggle */}
            <div className="flex items-center justify-center gap-4">
              <span className={`text-sm ${billingCycle === 'monthly' ? 'text-white' : 'text-slate-400'}`}>Monthly</span>
              <Switch
                checked={billingCycle === 'yearly'}
                onCheckedChange={(checked) => setBillingCycle(checked ? 'yearly' : 'monthly')}
                className="data-[state=checked]:bg-cyan-500"
              />
              <span className={`text-sm ${billingCycle === 'yearly' ? 'text-white' : 'text-slate-400'}`}>
                Yearly
                <Badge className="ml-2 bg-green-500/20 text-green-400 border-green-500/30 text-xs">Save 20%</Badge>
              </span>
            </div>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8"
          >
            {pricingPlans.map((plan, i) => (
              <motion.div
                key={i}
                variants={fadeInUp}
                whileHover={{ y: -10 }}
                className={`relative ${plan.popular ? 'md:-mt-4 md:mb-4' : ''}`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 z-10">
                    <Badge className="bg-gradient-to-r from-cyan-500 to-purple-500 text-white px-4 py-1">
                      <Rocket className="w-3 h-3 mr-1" />
                      Most Popular
                    </Badge>
                  </div>
                )}
                <Card className={`h-full bg-slate-800/30 backdrop-blur-sm transition-all duration-300 ${
                  plan.popular 
                    ? 'border-cyan-500/50 ring-2 ring-cyan-500/20' 
                    : 'border-white/5 hover:border-white/20'
                }`}>
                  <CardHeader className="text-center pb-2">
                    <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${plan.gradient} flex items-center justify-center mx-auto mb-4`}>
                      <plan.icon className="w-7 h-7 text-white" />
                    </div>
                    <CardTitle className="text-xl text-white">{plan.name}</CardTitle>
                    <CardDescription className="text-slate-400">{plan.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="text-center py-4">
                    <div className="mb-6">
                      <span className="text-4xl font-bold text-white">
                        ${plan.price[billingCycle]}
                      </span>
                      <span className="text-slate-400">/{billingCycle === 'monthly' ? 'mo' : 'mo'}</span>
                      {billingCycle === 'yearly' && plan.price.yearly > 0 && (
                        <div className="text-sm text-green-400 mt-1">
                          Billed ${plan.price.yearly * 12}/year
                        </div>
                      )}
                    </div>
                    
                    {/* Features List */}
                    <div className="space-y-3 text-left">
                      {plan.features.map((feature, j) => (
                        <div key={j} className="flex items-start gap-3">
                          {feature.included ? (
                            <div className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                              <Check className="w-3 h-3 text-green-400" />
                            </div>
                          ) : (
                            <div className="w-5 h-5 rounded-full bg-slate-700/50 flex items-center justify-center flex-shrink-0 mt-0.5">
                              <XCircle className="w-3 h-3 text-slate-500" />
                            </div>
                          )}
                          <span className={`text-sm ${feature.included ? 'text-slate-300' : 'text-slate-500'}`}>
                            {feature.text}
                          </span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                  <CardFooter className="pt-4">
                    <Button 
                      className={`w-full ${
                        plan.popular 
                          ? 'bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white' 
                          : 'bg-slate-700 hover:bg-slate-600 text-white'
                      }`}
                      onClick={() => plan.price.monthly === 0 ? openAuthModal('signup') : scrollToSection('pricing')}
                    >
                      {plan.cta}
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            ))}
          </motion.div>

          {/* Enterprise CTA */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeInUp}
            className="mt-12 text-center"
          >
            <p className="text-slate-400 mb-4">
              Need a custom solution for your healthcare organization?
            </p>
            <Button variant="outline" className="border-white/20 text-white hover:bg-white/10">
              Contact Our Sales Team
              <ChevronRight className="w-4 h-4 ml-2" />
            </Button>
          </motion.div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeInUp}
            className="text-center mb-12"
          >
            <Badge variant="outline" className="mb-4 border-pink-500/50 bg-pink-500/10 text-pink-400">
              Testimonials
            </Badge>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Trusted by Healthcare Professionals
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              See what doctors and patients are saying about MedVision.
            </p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {testimonials.map((testimonial, i) => (
              <motion.div
                key={i}
                variants={fadeInUp}
                whileHover={{ y: -5 }}
              >
                <Card className="h-full bg-slate-800/30 border-white/5 backdrop-blur-sm">
                  <CardContent className="pt-6">
                    <div className="flex mb-4">
                      {[...Array(testimonial.rating)].map((_, j) => (
                        <Star key={j} className="w-4 h-4 text-amber-400 fill-amber-400" />
                      ))}
                    </div>
                    <p className="text-slate-300 mb-4">&ldquo;{testimonial.content}&rdquo;</p>
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center text-white font-bold">
                        {testimonial.name.charAt(0)}
                      </div>
                      <div>
                        <div className="font-medium text-white">{testimonial.name}</div>
                        <div className="text-sm text-slate-400">{testimonial.role}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={scaleIn}
          >
            <Card className="relative overflow-hidden bg-gradient-to-br from-cyan-500/20 via-purple-500/20 to-pink-500/20 border-white/10 backdrop-blur-xl">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-purple-500/10" />
              <CardContent className="relative text-center py-12 px-6">
                <Sparkles className="w-12 h-12 mx-auto mb-6 text-cyan-400" />
                <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                  Start Your Health Journey Today
                </h2>
                <p className="text-slate-300 max-w-2xl mx-auto mb-8">
                  Join thousands of users who have taken control of their health with MedVision&apos;s 
                  AI-powered analysis and recommendations.
                </p>
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                  <Button 
                    size="lg" 
                    className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white px-8 h-14 text-lg border-0 group"
                    onClick={() => openAuthModal('signup')}
                  >
                    Get Started Free
                    <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                  </Button>
                  <div className="flex items-center gap-2 text-slate-400">
                    <Smartphone className="w-4 h-4" />
                    <span className="text-sm">Available on all devices</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 sm:px-6 lg:px-8 border-t border-white/10">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
            <div className="col-span-2 md:col-span-1">
              <div className="flex items-center gap-2 mb-4">
                <img 
                  src="/logo.png" 
                  alt="MedVision Logo" 
                  className="w-8 h-8 rounded-lg object-contain"
                />
                <span className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  MedVision
                </span>
              </div>
              <p className="text-sm text-slate-400">
                AI-powered health analysis for everyone.
              </p>
            </div>
            
            {[
              { title: 'Product', links: ['Features', 'Pricing', 'API', 'Integrations'] },
              { title: 'Company', links: ['About', 'Blog', 'Careers', 'Contact'] },
              { title: 'Legal', links: ['Privacy', 'Terms', 'HIPAA', 'Security'] },
            ].map((section, i) => (
              <div key={i}>
                <h4 className="font-semibold text-white mb-4">{section.title}</h4>
                <ul className="space-y-2">
                  {section.links.map((link) => (
                    <li key={link}>
                      <button 
                        onClick={() => link === 'Pricing' && scrollToSection('pricing')}
                        className="text-sm text-slate-400 hover:text-white transition-colors"
                      >
                        {link}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          
          <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-sm text-slate-400">
              © 2026 MedVision. All rights reserved.
            </p>
            <p className="text-xs text-slate-500">
              Disclaimer: This is for informational purposes only. Always consult healthcare professionals.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
