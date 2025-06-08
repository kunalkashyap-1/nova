// types/auth.ts
export interface User {
    id: number;
    email: string;
    username: string;
    full_name: string;
    bio?: string;
    preferred_language?: string;
    timezone?: string;
    profile_picture?: string;
    created_at: string;
  }
  
  export interface LoginRequest {
    username: string;
    password: string;
  }
  
  export interface RegisterRequest {
    full_name: string;
    email: string;
    username: string;
    password: string;
    bio?: string;
    preferred_language?: string;
    timezone?: string;
    profile_picture?: File;
  }
  
  export interface AuthResponse {
    user: User;
  }
  
  export interface AuthState {
    user: User | null;
    isAuthenticated: boolean;
    isLoading: boolean;
  }
  
  export interface AuthCheckResponse {
    authenticated: boolean;
    user?: User;
  }
  
  // UI Form interfaces
  export interface LoginFormData {
    username: string;
    password: string;
  }
  
  export interface RegisterFormData {
    fullName: string;
    email: string;
    username: string;
    password: string;
    profilePicture?: File;
    bio: string;
    preferredLanguage: string;
    timezone: string;
  }