<script lang="ts">
	import { Camera, User } from '@lucide/svelte';

	interface LoginFormData {
		username: string;
		password: string;
	}

	interface RegisterFormData {
		fullName: string;
		email: string;
		username: string;
		password: string;
		profilePicture?: File;
		bio: string;
		preferredLanguage: string;
		timezone: string;
	}

	let isLogin: boolean = $state(true);

	let loginFormData: LoginFormData = $state({
		username: '',
		password: ''
	});
	let registerFormData: RegisterFormData = $state({
		fullName: '',
		email: '',
		username: '',
		password: '',
		bio: '',
		preferredLanguage: '',
		timezone: ''
	});

	let profilePictureUrl: string = $state('');

	let fileInput: HTMLInputElement | null = $state(null);

	function handleLogin(event: Event): void {
		event.preventDefault();
		console.log('Login:', loginFormData);
	}

	function handleRegister(event: Event): void {
		event.preventDefault();
		console.log('Register:', registerFormData);
		// if (registerFormData.profilePicture) {
		// 	console.log('Profile Picture:', registerFormData.profilePicture);
		// }
	}

	function handleFileChange(event: Event): void {
		const input = event.target as HTMLInputElement;
		if (input.files && input.files.length > 0) {
			registerFormData.profilePicture = input.files[0];
			profilePictureUrl = URL.createObjectURL(registerFormData.profilePicture);
		}
	}

	function triggerFileInput(): void {
		fileInput?.click();
	}

</script>

<section class="flex min-h-screen items-center justify-center bg-gray-100 px-4">
	<div class="w-full max-w-lg space-y-8 rounded-lg bg-white p-8 shadow-md">
		{#if isLogin}
			<div class="space-y-6">
				<h1 class="text-center text-3xl font-bold text-gray-800">Login</h1>
				<form onsubmit={handleLogin} class="space-y-6">
					<div class="space-y-4">
						<div>
							<label for="login-username" class="block text-sm font-medium text-gray-700">
								Username <span class="text-red-500">*</span>
							</label>
							<input
								bind:value={loginFormData.username}
								id="login-username"
								type="text"
								required
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							/>
						</div>
						<div>
							<label for="login-password" class="block text-sm font-medium text-gray-700">
								Password <span class="text-red-500">*</span>
							</label>
							<input
								bind:value={loginFormData.password}
								id="login-password"
								type="password"
								required
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							/>
						</div>
					</div>
					<button
						type="submit"
						class="w-full rounded-md bg-blue-600 px-4 py-2 text-white shadow-sm hover:bg-blue-700 focus:outline-none"
					>
						Login
					</button>
				</form>
                <div class="flex items-center justify-around gap-2">
				<div class="flex items-center justify-center gap-2">
					<p class="text-sm text-gray-600">Don't have an account?</p>
					<button onclick={() => (isLogin = false)} class="text-sm text-blue-600 hover:underline">
						Register
					</button>
				</div>
                {@render tryIt()}
                </div>
			</div>
		{:else}
			<div class="space-y-6">
				<h1 class="text-center text-3xl font-bold text-gray-800">Register</h1>

				<div class="flex flex-col items-center">
					<div class="relative">
						<div
							class="group relative h-24 w-24 overflow-hidden rounded-full bg-gray-200 shadow-md"
						>
							{#if profilePictureUrl}
								<img
									src={profilePictureUrl}
									alt="Profile Preview"
									class="h-full w-full object-cover"
								/>
							{:else}
								<div class="flex h-full w-full items-center justify-center text-gray-500">
									<User size={40} />
								</div>
							{/if}
						</div>

						<button
							type="button"
							aria-label="Upload profile picture"
							onclick={triggerFileInput}
							class="absolute -right-1 bottom-0 flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-white shadow-md hover:bg-blue-700"
						>
							<Camera size={20} />
						</button>
					</div>

					<input
						type="file"
						accept="image/*"
						class="hidden"
						bind:this={fileInput}
						onchange={handleFileChange}
					/>
				</div>

				<form onsubmit={handleRegister} class="space-y-4">
					<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-700" for="register-fullName">
								Full Name <span class="text-red-500">*</span>
							</label>
							<input
								bind:value={registerFormData.fullName}
								id="register-fullName"
								type="text"
								required
								placeholder="Full name"
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							/>
						</div>

						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-700" for="register-email">
								Email Address <span class="text-red-500">*</span>
							</label>
							<input
								bind:value={registerFormData.email}
								id="register-email"
								type="email"
								required
								placeholder="you@example.com"
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							/>
						</div>

						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-700" for="register-username">
								Username <span class="text-red-500">*</span>
							</label>
							<input
								bind:value={registerFormData.username}
								id="register-username"
								type="text"
								required
								placeholder="Choose a username"
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							/>
						</div>

						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-700" for="register-password">
								Password <span class="text-red-500">*</span>
							</label>
							<input
								bind:value={registerFormData.password}
								id="register-password"
								type="password"
								required
								placeholder="Create a password"
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							/>
						</div>

						<div class="col-span-1">
							<label
								class="block text-sm font-medium text-gray-700"
								for="register-preferredLanguage"
							>
								Preferred Language
							</label>
							<input
								bind:value={registerFormData.preferredLanguage}
								id="register-preferredLanguage"
								type="text"
								placeholder="e.g., English"
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							/>
						</div>

						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-700" for="register-timezone">
								Timezone
							</label>
							<input
								bind:value={registerFormData.timezone}
								id="register-timezone"
								type="text"
								placeholder="e.g., America/New_York"
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							/>
						</div>

						<div class="col-span-1 md:col-span-2">
							<label class="block text-sm font-medium text-gray-700" for="register-bio">
								Bio
							</label>
							<textarea
								bind:value={registerFormData.bio}
								id="register-bio"
								rows="3"
								placeholder="Tell us about yourself"
								class="w-full rounded-md border px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500"
							></textarea>
						</div>
					</div>

					<button
						type="submit"
						class="w-full rounded-md bg-green-600 px-4 py-2 text-white shadow-sm hover:bg-green-700 focus:outline-none"
					>
						Register
					</button>
				</form>

                <div class="flex items-center justify-around gap-2">
				<div class="flex items-center justify-center gap-2">
					<p class="text-sm text-gray-600">Already have an account?</p>
					<button onclick={() => (isLogin = true)} class="text-sm text-blue-600 hover:underline">
						Login
					</button>
				</div>
                {@render tryIt()}
                </div>
			</div>
		{/if}
	</div>
</section>

{#snippet tryIt()}
	<div>
		<a href="/chat" class="text-sm text-purple-600 hover:text-purple-800 transition-colors">
			Try It Now
		</a>
	</div>
{/snippet}
