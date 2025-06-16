// lib/auth-api.ts
import type { 
    User, 
    LoginRequest, 
    RegisterRequest, 
    AuthCheckResponse 
  } from '../types/auth';
  import api from './api';
  import type { AxiosRequestConfig, AxiosResponse } from 'axios';
  
  class AuthAPI {
    private async makeRequest<T>(
      endpoint: string,
      options: AxiosRequestConfig = {}
    ): Promise<T> {
      try {
        const config: AxiosRequestConfig = {
          withCredentials: true, // Important for cookies
          ...options,
        };
  
        const response: AxiosResponse<T> = await api.request({
          url: endpoint,
          ...config,
        });
  
        return response.data;
      } catch (error: any) {
        // Handle axios errors
        const errorMessage = error.response?.data?.detail || 
                            error.response?.data?.message || 
                            error.message || 
                            'An error occurred';
        throw new Error(errorMessage);
      }
    }
  
    async login(credentials: LoginRequest): Promise<User> {
      return this.makeRequest<User>('/auth/login', {
        method: 'POST',
        data: credentials,
      });
    }
  
    async register(userData: RegisterRequest): Promise<User> {
      // Use FormData for file upload
      const formData = new FormData();
      formData.append('full_name', userData.full_name);
      formData.append('email', userData.email);
      formData.append('username', userData.username);
      formData.append('password', userData.password);
      
      if (userData.bio) formData.append('bio', userData.bio);
      if (userData.preferred_language) formData.append('preferred_language', userData.preferred_language);
      if (userData.timezone) formData.append('timezone', userData.timezone);
      if (userData.profile_picture) formData.append('profile_picture', userData.profile_picture);
  
      return this.makeRequest<User>('/auth/register', {
        method: 'POST',
        data: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    }
  
    async logout(): Promise<void> {
      await this.makeRequest<void>('/auth/logout', {
        method: 'POST',
      });
    }
  
    async getCurrentUser(): Promise<User> {
      return this.makeRequest<User>('/auth/me', {
        method: 'GET',
      });
    }
  
    async checkAuthStatus(): Promise<AuthCheckResponse> {
      return this.makeRequest<AuthCheckResponse>('/auth/check', {
        method: 'GET',
      });
    }
  
    async refreshToken(): Promise<void> {
      await this.makeRequest<void>('/auth/refresh', {
        method: 'POST',
      });
    }
  
    async updateProfile(userData: Partial<RegisterRequest>): Promise<User> {
      const formData = new FormData();
      
      if (userData.full_name) formData.append('full_name', userData.full_name);
      if (userData.bio) formData.append('bio', userData.bio);
      if (userData.preferred_language) formData.append('preferred_language', userData.preferred_language);
      if (userData.timezone) formData.append('timezone', userData.timezone);
      if (userData.profile_picture) formData.append('profile_picture', userData.profile_picture);
  
      return this.makeRequest<User>('/auth/me', {
        method: 'PUT',
        data: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    }
  }
  
  export const authAPI = new AuthAPI();