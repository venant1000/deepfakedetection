# DeepGuard AI Platform

## Overview
DeepGuard AI is a web application for deepfake detection and analysis. The platform allows users to upload video files, analyze them for potential deepfake manipulation, and view detailed analysis results. The application features user authentication, a dashboard for tracking analyses, and detailed reports on detected deepfakes.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: React with TypeScript
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: React Query for server state management
- **UI Components**: ShadCN UI component library with Tailwind CSS styling
- **Form Handling**: React Hook Form with Zod for validation

### Backend
- **Framework**: Express.js with TypeScript
- **Authentication**: Passport.js with local strategy
- **API**: RESTful API endpoints for user management and video analysis
- **File Handling**: Multer for handling video file uploads

### Database
- **ORM**: Drizzle ORM for database interactions
- **Schema**: PostgreSQL database schema defined in `shared/schema.ts`
- **Tables**: 
  - Users (authentication and profile data)
  - VideoAnalyses (storage for analysis results)

### Development & Build
- **Development Server**: Vite for frontend development
- **Build System**: ESBuild for server bundling, Vite for frontend bundling
- **Deployment**: Configured for Replit deployment via `.replit` file

## Key Components

### Authentication System
- Passport.js integration with session-based authentication
- Password hashing using scrypt for security
- Login/Registration forms with validation

### Video Analysis Pipeline
- File upload handling with size and type validation
- Analysis processing for deepfake detection
- Results storage and retrieval system

### User Interface
1. **Landing Page**: Marketing-focused page with feature highlights
2. **Authentication Page**: Login and registration forms
3. **Dashboard**: Overview of user's analyses with statistics
4. **Analysis Page**: Detailed view of individual analysis results
5. **Admin Page**: Platform management for administrators

### Shared Schema
The application uses a shared schema between frontend and backend, defined in TypeScript, ensuring type safety across the stack. This includes:
- User types and validation
- Video analysis data structures
- Form validation schemas

## Data Flow

1. **Authentication Flow**:
   - User submits credentials
   - Server validates and creates session
   - Client stores session cookie
   - Protected routes check authentication status

2. **Video Analysis Flow**:
   - User uploads video file from dashboard
   - Server receives file and validates size/type
   - Analysis is performed (integration point for AI detection services)
   - Results are stored in database
   - User is redirected to analysis details page

3. **Data Fetching Flow**:
   - React Query handles API requests and caching
   - API endpoints serve data with proper authentication checks
   - UI components display data with loading/error states

## External Dependencies

### Frontend
- React ecosystem (React, React DOM)
- Wouter for routing
- React Query for data fetching
- Radix UI components (via ShadCN UI)
- Tailwind CSS for styling
- Zod for validation
- Lucide React for icons

### Backend
- Express for API server
- Passport for authentication
- Multer for file uploads
- Drizzle ORM for database interactions

## Deployment Strategy

The application is configured for deployment on Replit with:
- Production build process: `npm run build`
- Start command: `npm run start`
- Port configuration: 5000 (internal) to 80 (external)
- Database: PostgreSQL configured via Replit's database module

### Development Workflow
1. Run `npm run dev` for local development
2. Database migrations managed via Drizzle Kit
3. Type checking via TypeScript

### Production Build
1. Frontend built with Vite
2. Backend bundled with ESBuild
3. Deployable as a single package

## Future Integration Points

- **AI Model Integration**: The codebase is structured to easily integrate deepfake detection AI models
- **Additional Authentication Methods**: Structure allows for adding OAuth providers
- **Analytics Expansion**: Admin dashboard can be extended with more detailed analytics