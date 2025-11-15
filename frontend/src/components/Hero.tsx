import { Button } from "./ui/button";
import { ArrowRight, Play, Sparkles, CheckCircle2 } from "lucide-react";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { Badge } from "./ui/badge";
import { motion } from "framer-motion";
import { FloatingElements } from "./FloatingElements";

export function Hero() {
  return (
    <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      <FloatingElements />
      
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div 
          className="absolute top-20 right-20 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3]
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div 
          className="absolute bottom-20 left-20 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.3, 0.6, 0.3]
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1
          }}
        />
      </div>

      <div className="container mx-auto relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <motion.div 
            className="space-y-8"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Badge className="inline-flex items-center gap-2 bg-gradient-to-r from-blue-50 to-purple-50 text-blue-700 px-4 py-2 rounded-full border border-blue-200/50 hover:shadow-md transition-shadow">
                <Sparkles className="h-4 w-4" />
                <span className="text-sm">AI-Powered OCR Technology</span>
              </Badge>
            </motion.div>
            
            <motion.h1 
              className="text-5xl sm:text-6xl lg:text-7xl bg-gradient-to-r from-slate-900 via-blue-800 to-purple-800 bg-clip-text text-transparent leading-tight"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              Transform Documents into Data Instantly
            </motion.h1>
            
            <motion.p 
              className="text-xl text-slate-600 max-w-xl leading-relaxed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              Automate your document processing with our cutting-edge OCR technology. 
              Extract text, tables, and data from any document in seconds with 99.9% accuracy.
            </motion.p>

            <motion.div 
              className="space-y-3"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <div className="flex items-center gap-2 text-slate-700">
                <CheckCircle2 className="h-5 w-5 text-green-600" />
                <span>No credit card required for 14-day trial</span>
              </div>
              <div className="flex items-center gap-2 text-slate-700">
                <CheckCircle2 className="h-5 w-5 text-green-600" />
                <span>Process unlimited documents during trial</span>
              </div>
              <div className="flex items-center gap-2 text-slate-700">
                <CheckCircle2 className="h-5 w-5 text-green-600" />
                <span>Enterprise-grade security & compliance</span>
              </div>
            </motion.div>

            <motion.div 
              className="flex flex-col sm:flex-row gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button 
                  size="lg" 
                  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-lg h-14 px-8 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 transition-all"
                >
                  Start Free Trial
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button 
                  size="lg" 
                  variant="outline" 
                  className="text-lg h-14 px-8 border-2 border-slate-300 hover:border-slate-400 hover:bg-slate-50"
                >
                  <Play className="mr-2 h-5 w-5" />
                  Watch Demo
                </Button>
              </motion.div>
            </motion.div>

            <motion.div 
              className="flex items-center gap-6 pt-4 flex-wrap"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              <div className="flex items-center gap-2">
                <div className="flex -space-x-2">
                  <motion.div 
                    className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 border-2 border-white"
                    whileHover={{ scale: 1.2, zIndex: 10 }}
                  />
                  <motion.div 
                    className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-400 to-purple-600 border-2 border-white"
                    whileHover={{ scale: 1.2, zIndex: 10 }}
                  />
                  <motion.div 
                    className="w-8 h-8 rounded-full bg-gradient-to-br from-pink-400 to-pink-600 border-2 border-white"
                    whileHover={{ scale: 1.2, zIndex: 10 }}
                  />
                </div>
                <span className="text-sm text-slate-600">Trusted by 10,000+ companies</span>
              </div>
            </motion.div>
          </motion.div>

          <motion.div 
            className="relative"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.div 
              className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur-3xl opacity-20"
              animate={{
                scale: [1, 1.1, 1],
                rotate: [0, 5, 0]
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
            <motion.div 
              className="relative bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden"
              whileHover={{ 
                scale: 1.02,
                rotateY: 5,
                rotateX: 5
              }}
              style={{ transformStyle: "preserve-3d" }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <ImageWithFallback 
                src="https://images.unsplash.com/photo-1658142034594-12b5c411069d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkb2N1bWVudCUyMHNjYW5uaW5nJTIwdGVjaG5vbG9neXxlbnwxfHx8fDE3NjMwMzc4MTN8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                alt="OCR Technology"
                className="w-full h-auto"
              />
              {/* Floating stats badge */}
              <motion.div 
                className="absolute bottom-6 left-6 right-6 bg-white/95 backdrop-blur-sm rounded-xl shadow-lg p-4 border border-slate-200"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 }}
                whileHover={{ scale: 1.05 }}
              >
                <div className="grid grid-cols-3 gap-4">
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                  >
                    <div className="text-2xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">99.9%</div>
                    <div className="text-xs text-slate-600">Accuracy</div>
                  </motion.div>
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                  >
                    <div className="text-2xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">10M+</div>
                    <div className="text-xs text-slate-600">Docs/Month</div>
                  </motion.div>
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                  >
                    <div className="text-2xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">{"<2s"}</div>
                    <div className="text-xs text-slate-600">Avg. Time</div>
                  </motion.div>
                </div>
              </motion.div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}